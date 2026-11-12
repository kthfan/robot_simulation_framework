"""CPU collision checking with coal (hpp-fcl): exact triangle meshes and primitives.

The robot's collision geometry comes from its URDF (mesh links become BVH
triangle meshes; meshes that are exact boxes become boxes). Obstacles are a
World's solids. Each configuration is checked by

1. forward kinematics of every link (:meth:`UrdfModel.fk`);
2. a bounding-sphere broad phase that drops pairs farther apart than
   ``cutoff``;
3. coal's signed distance for the rest (negative is penetration depth);
4. the :class:`~rsf.collision.ContactPolicy` margin of each pair.

Attached objects (a carried tray) are extra geometry fixed to a link and are
matched by rules under the link name ``attached/<name>``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from rsf.collision import ConfigReport, ContactPolicy, PairResult
from rsf.model.urdf import UrdfModel
from rsf.model.world import Shape, Solid, solids_from_urdf


def _coal():
    import coal
    return coal


_MESH_CACHE: dict = {}


def _load_mesh(path: Path, scale=None):
    """``(vertices, faces)`` of a mesh file."""
    import trimesh
    key = (str(path), None if scale is None else tuple(np.asarray(scale, float)))
    if key not in _MESH_CACHE:
        m = trimesh.load(path, force="mesh", process=False)
        V = np.asarray(m.vertices, dtype=float)
        if scale is not None:
            V = V * np.asarray(scale, float)
        _MESH_CACHE[key] = (V, np.asarray(m.faces, dtype=np.int64))
    return _MESH_CACHE[key]


@dataclass
class _Geom:
    name: str
    geom: object             #: coal collision geometry
    T_local: np.ndarray      #: in the owning frame (link, or world for obstacles)
    centre: np.ndarray       #: bounding sphere centre, in the geometry frame
    radius: float
    solid: Optional[Solid] = None


def make_geometry(shape: Shape):
    """``(coal geometry, sphere centre, sphere radius)`` for a :class:`Shape`."""
    coal = _coal()
    if shape.kind == "box":
        s = np.asarray(shape.size, float)
        return coal.Box(*s), np.zeros(3), float(np.linalg.norm(s) / 2.0)
    if shape.kind == "mesh":
        V, F = _load_mesh(shape.path, shape.scale)
        m = coal.BVHModelOBBRSS()
        m.beginModel(len(F), len(V))
        m.addVertices(V)
        m.addTriangles(F)
        m.endModel()
        c = (V.min(0) + V.max(0)) / 2.0
        return m, c, float(np.linalg.norm(V - c, axis=1).max())
    if shape.kind == "cylinder":
        g = coal.Cylinder(float(shape.radius), float(shape.length))
        return g, np.zeros(3), float(np.hypot(shape.radius, shape.length / 2.0))
    if shape.kind == "sphere":
        return coal.Sphere(float(shape.radius)), np.zeros(3), float(shape.radius)
    raise ValueError(f"unsupported shape {shape.kind!r}")


def _tf(T):
    coal = _coal()
    t = coal.Transform3s()
    t.setRotation(np.ascontiguousarray(T[:3, :3]))
    t.setTranslation(np.ascontiguousarray(T[:3, 3]))
    return t


def signed_distance(g1, T1, g2, T2) -> float:
    coal = _coal()
    return float(coal.distance(g1, _tf(T1), g2, _tf(T2), coal.DistanceRequest(), coal.DistanceResult()))


# ─────────────────────────────────────────────────────────────────────────────
# Self-collision pairs
# ─────────────────────────────────────────────────────────────────────────────
def rigid_bodies(model: UrdfModel) -> dict:
    """``link -> body``: links joined by fixed joints share a body (its top link)."""
    body = {}

    def walk(link, b):
        body[link] = b
        for j in model.children.get(link, []):
            walk(j.child, b if j.type == "fixed" else j.child)

    walk(model.root, model.root)
    return body


def candidate_self_pairs(model: UrdfModel) -> list:
    """Link pairs on different, non-adjacent bodies, both with collision geometry."""
    body = rigid_bodies(model)
    adjacent = {frozenset((body[j.parent], body[j.child])) for j in model.joints.values() if j.type != "fixed"}
    links = [n for n, l in model.links.items() if l.collisions]
    out = []
    for a, b in itertools.combinations(links, 2):
        ba, bb = body[a], body[b]
        if ba != bb and frozenset((ba, bb)) not in adjacent:
            out.append((a, b))
    return out


# ─────────────────────────────────────────────────────────────────────────────
class CoalChecker:
    """Exact distances between a robot (URDF) and a set of world solids."""

    def __init__(self, urdf, base, policy: ContactPolicy, *, self_pairs: Optional[Iterable] = None,
                 self_margin: float = 0.0, cutoff: float = 0.05):
        self.model = urdf if isinstance(urdf, UrdfModel) else UrdfModel.from_file(urdf)
        self.base = np.asarray(base, dtype=float)
        self.policy = policy
        self.cutoff = cutoff
        self.self_margin = self_margin
        self.links: dict[str, list] = {}
        for s in solids_from_urdf(self.model):
            link = s.name.split("/", 1)[0] if "/" in s.name else self.model.root
            g, c, r = make_geometry(s.shape)
            # solids_from_urdf poses are at q = 0 in the root frame: keep the
            # pose relative to the link instead.
            self.links.setdefault(link, []).append((s, g, c, r))
        fk0 = self.model.fk()
        self._link_geoms = {
            link: [_Geom(s.name, g, np.linalg.inv(fk0[link]) @ s.T, c, r) for s, g, c, r in items]
            for link, items in self.links.items()}
        self.self_pairs = list(candidate_self_pairs(self.model) if self_pairs is None else self_pairs)
        self.attached: dict[str, tuple] = {}
        self.obstacles: list[_Geom] = []
        self._rule_cache: dict = {}

    # ── world ──────────────────────────────────────────────────────────────
    def set_obstacles(self, solids: Iterable[Solid]) -> None:
        """Replace the obstacles. Geometry is built once per shape and reused."""
        self.obstacles = []
        cache = self.__dict__.setdefault("_shape_cache", {})
        for s in solids:
            key = id(s.shape)
            if key not in cache:
                cache[key] = (s.shape,) + tuple(make_geometry(s.shape))   # keep the shape alive
            _, g, c, r = cache[key]
            self.obstacles.append(_Geom(s.name, g, np.asarray(s.T, float), c, r, s))
        self._obs_c = np.array([(o.T_local[:3, :3] @ o.centre + o.T_local[:3, 3]) for o in self.obstacles]) \
            if self.obstacles else np.zeros((0, 3))
        self._obs_r = np.array([o.radius for o in self.obstacles])
        self._rule_cache.clear()

    def attach(self, name: str, link: str, solids: Iterable[Solid], T_link_object) -> None:
        """Fix *solids* (in the object frame) to *link* at *T_link_object*."""
        T = np.asarray(T_link_object, float)
        geoms = []
        for s in solids:
            g, c, r = make_geometry(s.shape)
            geoms.append(_Geom(s.name, g, T @ s.T, c, r, s))
        self.attached[name] = (link, geoms)
        self._rule_cache.clear()

    def detach(self, name: str) -> None:
        self.attached.pop(name, None)
        self._rule_cache.clear()

    # ── queries ────────────────────────────────────────────────────────────
    def link_poses(self, q: dict) -> dict:
        return {k: self.base @ v for k, v in self.model.fk(q).items()}

    def _bodies(self, poses: dict):
        """``(rule_name, link, geom, world pose)`` for every robot and attached geometry."""
        for link, geoms in self._link_geoms.items():
            for g in geoms:
                yield link, link, g, poses[link] @ g.T_local
        for name, (link, geoms) in self.attached.items():
            for g in geoms:
                yield f"attached/{name}", link, g, poses[link] @ g.T_local

    def _rule(self, who: str, obs: _Geom):
        key = (who, obs.name)
        if key not in self._rule_cache:
            self._rule_cache[key] = self.policy.resolve(who, obs.solid)
        return self._rule_cache[key]

    def check(self, q: dict, *, group: Optional[Callable] = None, self_collision: bool = True) -> ConfigReport:
        """Every robot/attached geometry against the obstacles (and itself)."""
        poses = self.link_poses(q)
        results = []
        for who, link, g, T in self._bodies(poses):
            if not len(self.obstacles):
                break
            c = T[:3, :3] @ g.centre + T[:3, 3]
            gap = np.linalg.norm(self._obs_c - c, axis=1) - self._obs_r - g.radius
            for i in np.nonzero(gap < self.cutoff)[0]:
                o = self.obstacles[i]
                rule = self._rule(who, o)
                if rule is None:
                    continue
                d = signed_distance(g.geom, T, o.geom, o.T_local)
                results.append(PairResult(who, o.name, d, rule[0], rule[1]))
        self_v = []
        if self_collision:
            for a, b in self.self_pairs:
                for ga in self._link_geoms.get(a, []):
                    Ta = poses[a] @ ga.T_local
                    ca = Ta[:3, :3] @ ga.centre + Ta[:3, 3]
                    for gb in self._link_geoms.get(b, []):
                        Tb = poses[b] @ gb.T_local
                        cb = Tb[:3, :3] @ gb.centre + Tb[:3, 3]
                        if np.linalg.norm(ca - cb) - ga.radius - gb.radius >= self.self_margin + 1e-3:
                            continue
                        d = signed_distance(ga.geom, Ta, gb.geom, Tb)
                        if d < self.self_margin:
                            self_v.append(PairResult(a, b, d, self.self_margin, "self"))
        viol = [r for r in results if r.slack < 0.0]
        worst = min(results, key=lambda r: r.slack) if results else None
        near = {}
        if group is not None:
            for r in results:
                k = group(r.obstacle)
                if k not in near or r.distance < near[k].distance:
                    near[k] = r
        return ConfigReport(not viol and not self_v, worst, viol, self_v, near)

    def distance_to(self, q: dict, solids: Iterable[Solid], who: Optional[str] = None) -> float:
        """Smallest signed distance from the robot (or one link / attached name) to *solids*."""
        poses = self.link_poses(q)
        best = np.inf
        objs = [(make_geometry(s.shape)[0], s.T) for s in solids]
        for name, link, g, T in self._bodies(poses):
            if who is not None and name != who:
                continue
            for og, oT in objs:
                best = min(best, signed_distance(g.geom, T, og, oT))
        return best

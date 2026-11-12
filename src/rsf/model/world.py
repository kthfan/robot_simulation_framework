"""The compiled cell: solids, named frames and the robot, all in world coordinates.

A :class:`World` is what every backend consumes (collision checkers, simulators,
visualisers, planners). It is built from a layout (:mod:`rsf.model.layout`) and
never edited by hand, so the backends cannot disagree about the cell.

* :class:`Solid` — one convex-ish collision/visual primitive with a world pose.
* :class:`Feature` — a named frame with attributes (a slot, a label, a mount
  point); workflows select targets among features.
* :class:`Entity` — a named group of solids and features (a rack, a jig).
* :class:`ObjectInstance` — a movable object (a tray) and where it sits.

World frame: z up, metres.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from rsf.core import se3


@dataclass(frozen=True, eq=False)
class Shape:
    """Geometry in its own frame. ``box``: full ``size``; ``mesh``: ``path``
    (with ``scale``); ``cylinder``: ``radius``, ``length`` along z; ``sphere``."""
    kind: str
    size: Optional[np.ndarray] = None
    path: Optional[Path] = None
    scale: Optional[np.ndarray] = None
    radius: Optional[float] = None
    length: Optional[float] = None

    @staticmethod
    def box(size) -> "Shape":
        return Shape("box", size=np.asarray(size, dtype=float))

    @staticmethod
    def mesh(path, scale=None) -> "Shape":
        return Shape("mesh", path=Path(path), scale=None if scale is None else np.asarray(scale, float))


@dataclass(frozen=True, eq=False)
class Solid:
    name: str
    T: np.ndarray            #: world pose of the shape frame (box: its centre)
    shape: Shape
    entity: str = ""
    tags: frozenset = frozenset()

    @property
    def size(self) -> np.ndarray:
        return self.shape.size

    def corners(self) -> np.ndarray:
        if self.shape.kind != "box":
            raise ValueError(f"{self.name}: corners of a {self.shape.kind}")
        h = self.shape.size / 2.0
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
        return c @ self.T[:3, :3].T + self.T[:3, 3]

    def moved(self, T_world_new_parent) -> "Solid":
        return replace(self, T=np.asarray(T_world_new_parent, float) @ self.T)


@dataclass(frozen=True, eq=False)
class Feature:
    """A named frame. ``attrs`` are plain data (numbers, strings, lists)."""
    key: str
    T: np.ndarray
    kind: str
    entity: str = ""
    attrs: dict = field(default_factory=dict)
    tags: frozenset = frozenset()

    def get(self, name: str, default=None):
        if name in ("key", "kind", "entity"):
            return getattr(self, name)
        if name in ("x", "y", "z"):
            return float(self.T["xyz".index(name), 3])
        return self.attrs.get(name, default)


@dataclass(frozen=True, eq=False)
class Entity:
    name: str
    kind: str
    solids: tuple = ()
    features: tuple = ()
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class ObjectInstance:
    """A movable object: its solids are in the object frame; ``at`` names the
    feature it rests on (or None when carried); ``T`` is its world pose."""
    name: str
    kind: str
    solids: tuple
    T: np.ndarray
    at: Optional[str] = None
    mass: Optional[float] = None
    source: Optional[Path] = None

    def world_solids(self) -> list:
        """The object's solids in the world, named ``<object>/<solid>``."""
        return [replace(s.moved(self.T), name=f"{self.name}/{s.name}", entity=self.name) for s in self.solids]


@dataclass(frozen=True, eq=False)
class RobotSpec:
    urdf: Path
    base: np.ndarray          #: world pose of the URDF root link
    tool_frame: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class World:
    name: str
    entities: tuple
    objects: tuple = ()
    robot: Optional[RobotSpec] = None
    process: dict = field(default_factory=dict)   #: cell-level process parameters
    source: dict = field(default_factory=dict)    #: the resolved layout it came from

    # ── lookups ────────────────────────────────────────────────────────────
    def entity(self, name: str) -> Entity:
        for e in self.entities:
            if e.name == name:
                return e
        raise KeyError(f"no entity {name!r}")

    def features(self, select=None) -> list:
        from rsf.model.select import select_features
        out = [f for e in self.entities for f in e.features]
        return out if select is None else select_features(out, select)

    def feature(self, key: str) -> Feature:
        for e in self.entities:
            for f in e.features:
                if f.key == key:
                    return f
        raise KeyError(f"no feature {key!r}")

    def object_at(self, feature_key: str) -> Optional[ObjectInstance]:
        for o in self.objects:
            if o.at == feature_key:
                return o
        return None

    # ── geometry ───────────────────────────────────────────────────────────
    def solids(self, *, objects: bool = True, exclude_objects=()) -> list:
        out = [s for e in self.entities for s in e.solids]
        if objects:
            skip = set(exclude_objects)
            for o in self.objects:
                if o.name not in skip:
                    out += o.world_solids()
        return out

    def occupancy(self) -> dict:
        return {o.at: o.name for o in self.objects if o.at is not None}

    # ── state changes (return new worlds) ──────────────────────────────────
    def without_objects(self, names: Iterable[str]) -> "World":
        names = set(names)
        return replace(self, objects=tuple(o for o in self.objects if o.name not in names))

    def with_object(self, obj: ObjectInstance) -> "World":
        return replace(self, objects=tuple(o for o in self.objects if o.name != obj.name) + (obj,))


# ─────────────────────────────────────────────────────────────────────────────
# URDF links as solids
# ─────────────────────────────────────────────────────────────────────────────
def _obj_vertices(path: Path) -> Optional[np.ndarray]:
    if path.suffix.lower() != ".obj":
        return None
    v = [ln.split()[1:4] for ln in path.read_text().splitlines() if ln.startswith("v ")]
    return np.array(v, dtype=float) if v else None


def _as_box(path: Path, scale) -> Optional[tuple]:
    """``(T_box, size)`` if the mesh is exactly a rectangular cuboid (8 corners, any
    orientation): ``T_box`` puts the box centre and axes in the mesh frame."""
    V = _obj_vertices(path)
    if V is None or len(V) != 8:
        return None
    if scale is not None:
        V = V * np.asarray(scale, float)
    import itertools
    d = (V - V[0])[1:]
    found = None
    for i, j, k in itertools.combinations(range(7), 3):
        E = d[[i, j, k]]
        L = np.linalg.norm(E, axis=1)
        if np.any(L < 1e-9):
            continue
        U = E / L[:, None]
        if max(abs(U[0] @ U[1]), abs(U[0] @ U[2]), abs(U[1] @ U[2])) > 1e-6:
            continue
        want = np.array([V[0] + a * E[0] + b * E[1] + c * E[2] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        dist = np.linalg.norm(want[:, None, :] - V[None, :, :], axis=2)
        if dist.min(axis=1).max() <= 1e-7 * max(1.0, L.max()) and len(set(dist.argmin(axis=1))) == 8:
            found = (U, L)
            break
    if found is None:
        return None
    U, L = found
    # canonical axes: each box axis takes the frame axis it is closest to, pointing positive
    order = np.argmax(np.abs(U), axis=1)
    if len(set(order)) == 3:
        perm = np.argsort(order)
        U, L = U[perm], L[perm]
        U = U * np.sign(U[np.arange(3), np.arange(3)])[:, None]
    R = U.T.copy()
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return se3.make_T(R, V.mean(axis=0)), L.copy()


def solids_from_urdf(model, *, prefix: str = "", entity: str = "", tags=frozenset(),
                     boxes: bool = True, q=None) -> list:
    """Collision geometry of every link at joint values *q*, in the URDF root frame.

    With *boxes*, meshes that are exactly axis-aligned boxes become box shapes
    (exact, and cheaper for every backend). Solid names are
    ``<prefix><link>/<geometry name or index>``.
    """
    fk = model.fk(q)
    out = []
    for link in model.links.values():
        for i, g in enumerate(link.collisions):
            gname = g.name or str(i)
            T = fk[link.name] @ g.origin
            if g.kind == "box":
                shape = Shape.box(g.size)
            elif g.kind == "mesh":
                b = _as_box(g.path, g.scale) if boxes and g.path is not None else None
                if b is not None:
                    T = T @ b[0]
                    shape = Shape.box(b[1])
                else:
                    shape = Shape.mesh(g.path, g.scale)
            elif g.kind == "cylinder":
                shape = Shape("cylinder", radius=g.radius, length=g.length)
            else:
                shape = Shape("sphere", radius=g.radius)
            name = f"{prefix}{link.name}/{gname}" if len(model.links) > 1 else f"{prefix}{gname}"
            out.append(Solid(name, T, shape, entity, frozenset(tags)))
    return out

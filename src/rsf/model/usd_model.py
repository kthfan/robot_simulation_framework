"""USD files as collision geometry: a composite part (many meshes and their relative
placement) for layouts, like a URDF without joints.

::

    placements:
      jig: {model: ../assets/jig.usd, prim: /jig, at: {kind: slot}}
    objects:
      tray: {model: ../assets/tray.usda, mass: 1.2, at: {kind: slot}}
    entities:
      rack: {model: ../assets/rack.usdc, pose: {...}, frames: {prims: "/rack/frames/*", kind: slot}}

:func:`load` reads a stage with ``pxr`` (``pip install usd-core``, or Isaac Sim's own) and
returns :class:`UsdModel`: solids in the frame of ``prim`` (default: the stage's own frame),
the mass, and named frames.

**Units and axes.** ``metersPerUnit`` scales everything to metres; a ``Y``-up stage is
turned so its +Y is +Z (a turn of +90 deg about x). A stage that does not author them
gets USD's fallbacks: **centimetres and Y up** (Isaac Sim and most CAD exporters author
both; check ``UsdModel.meters_per_unit`` / ``up_axis`` when a part comes out 100x off).
References keep their own numbers: USD does not rescale a referenced layer with other
units, so a stage composed from files of mixed units is read in the root stage's. Poses in rsf are then as everywhere:
metres, z up. The part's frame is ``prim``'s frame, converted the same way.

**Which prims collide** (the first rule that selects anything wins):

1. ``collision: [glob, ...]``: prims whose USD path (or an ancestor's) matches a glob,
   e.g. ``/jig/collision/*``;
2. prims with ``UsdPhysics.CollisionAPI`` (``physics:collisionEnabled`` not false), as
   Isaac Sim and most physics exports mark them;
3. every geometric prim, except ``guide`` purpose; if any has ``proxy`` purpose, the
   ``render`` ones are left out (the proxy stands for them).

Always: instanced prims are read through their prototypes; ``guide`` prims never collide.

**Shapes.** ``Mesh`` (polygons fan-triangulated; ``leftHanded`` and mirrored transforms
keep the faces outward); ``Cube``; ``Sphere`` and ``Cylinder`` as exact shapes when their
scale is uniform (a cylinder: uniform across its axis is not enough, all three); otherwise,
and ``Capsule`` / ``Cone`` always, triangulated. A mesh's scale (and any shear) is baked
into its vertices; its solid's pose is the rigid part. A mesh that is exactly a box becomes
a box solid (as from URDFs). Other meshes are written once as OBJ files into the mesh
cache (``RSF_CACHE``, default ``~/.cache/rsf``), named by their content, because every
backend reads meshes from files.

**Names.** A solid is named by its prim path relative to ``prim``
(``Geometry/Cube_01``); the layout prefixes it as usual.

**Mass**: ``UsdPhysics.MassAPI``'s ``physics:mass`` on ``prim``, else the sum over the
rigid bodies under it (``kilogramsPerUnit`` applied); ``None`` without one (the layout's
``mass:`` overrides it).

**Looks**: :attr:`UsdModel.visual` (:func:`visual_source`) says where the part's
appearance is -- the file, ``prim`` and the stage-to-part conversion -- so the Isaac stage
(``rsf.compile.usd``) can reference the original prims for rendering only.

**Frames**: ``frames: {prims: <glob>, kind: <feature kind>}`` turns the matching prims
(any type) into named frames: name = the prim's name, attributes = its ``customData``
under ``rsf`` (``customData = {dictionary rsf = {string role = "source"}}``).

MuJoCo and PhysX collide a mesh as its convex hull; the planner (coal) uses the exact
triangles. Split concave parts into convex pieces in the USD; the audit's
``rsf.convex_meshes`` warns about meshes whose hull is noticeably larger.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from rsf.core import se3
from rsf.model.world import Shape, Solid, box_from_vertices

GPRIMS = ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")


class UsdError(ValueError):
    pass


@dataclass
class UsdModel:
    source: Path
    solids: list                         #: Solid, in the part's frame
    mass: Optional[float] = None
    frames: dict = field(default_factory=dict)   #: name -> (T, attrs)
    meters_per_unit: float = 1.0
    up_axis: str = "Z"
    rule: str = ""                       #: which collision rule selected the solids
    visual: Optional[dict] = None        #: where its looks are (:func:`visual_source`)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh cache
# ─────────────────────────────────────────────────────────────────────────────
def cache_dir() -> Path:
    return Path(os.environ.get("RSF_CACHE") or Path.home() / ".cache" / "rsf") / "meshes"


def write_obj(V: np.ndarray, F: np.ndarray, directory: Optional[Path] = None) -> Path:
    """*V*, *F* as an OBJ file named by its content (written once; safe to share)."""
    V = np.ascontiguousarray(V, dtype=float)
    F = np.ascontiguousarray(F, dtype=np.int64)
    h = hashlib.sha1(V.tobytes() + F.tobytes()).hexdigest()[:20]
    d = Path(directory) if directory is not None else cache_dir()
    path = d / f"{h}.obj"
    if not path.exists():
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"v {x!r} {y!r} {z!r}" for x, y, z in V.tolist()]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F.tolist()]
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _gf(m) -> np.ndarray:
    """A ``Gf.Matrix4d`` (row vectors) as a column-convention 4x4 array."""
    return np.array(m, dtype=float).T


def _split(M: np.ndarray) -> tuple:
    """``(T_rigid, P)``: ``M[:3,:3] = R P`` with R a rotation (polar decomposition); a
    mirror (det < 0) is put into P."""
    A = M[:3, :3]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    P = R.T @ A
    return se3.make_T(R, M[:3, 3]), P


def _uniform(P: np.ndarray, tol: float = 1e-9) -> Optional[float]:
    s = float(np.cbrt(np.linalg.det(P))) if np.linalg.det(P) > 0 else None
    if s is None or not np.allclose(P, s * np.eye(3), atol=tol * max(1.0, s)):
        return None
    return s


#: rotation taking z to a prim's ``axis`` (cylinders, capsules, cones)
AXIS = {"Z": np.eye(3), "X": se3.rot_y(math.pi / 2), "Y": se3.rot_x(-math.pi / 2)}


def _triangulate(counts, indices) -> np.ndarray:
    out, k = [], 0
    for n in counts:
        n = int(n)
        if n >= 3:
            f = indices[k:k + n]
            out += [(f[0], f[i], f[i + 1]) for i in range(1, n - 1)]
        k += n
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def _prim_mesh(prim) -> tuple:
    """``(V, F, exact)``: the prim's geometry in its own frame, or an exact shape
    ``exact = (kind, params)`` for spheres and cylinders."""
    from pxr import UsdGeom
    t = prim.GetTypeName()
    if t == "Mesh":
        m = UsdGeom.Mesh(prim)
        V = np.array(m.GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
        F = _triangulate(m.GetFaceVertexCountsAttr().Get() or [], list(m.GetFaceVertexIndicesAttr().Get() or []))
        if (m.GetOrientationAttr().Get() or "rightHanded") == "leftHanded":
            F = F[:, ::-1]
        return V, F, None
    if t == "Cube":
        h = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
        V = np.array([(x, y, z) for z in (-h, h) for y in (-h, h) for x in (-h, h)], float)
        F = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
                      (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)], np.int64)
        return V, F, None
    import trimesh
    if t == "Sphere":
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        tm = trimesh.creation.icosphere(subdivisions=3, radius=r)
        return np.asarray(tm.vertices), np.asarray(tm.faces), ("sphere", {"radius": r})
    g = {"Cylinder": UsdGeom.Cylinder, "Capsule": UsdGeom.Capsule, "Cone": UsdGeom.Cone}[t](prim)
    r = float(g.GetRadiusAttr().Get() or 1.0)
    h = float(g.GetHeightAttr().Get() or 2.0)
    axis = str(g.GetAxisAttr().Get() or "Z")
    Ra = AXIS[axis]
    if t == "Cylinder":
        tm = trimesh.creation.cylinder(radius=r, height=h, sections=48)
        exact = ("cylinder", {"radius": r, "length": h, "R": Ra})
    elif t == "Capsule":
        tm = trimesh.creation.capsule(height=h, radius=r, count=[24, 24])
        exact = None
    else:
        tm = trimesh.creation.cone(radius=r, height=h, sections=48)
        tm.apply_translation((0.0, 0.0, -h / 2.0))          # USD's cone is centred
        exact = None
    V = np.asarray(tm.vertices, float) @ Ra.T
    return V, np.asarray(tm.faces, np.int64), exact


def _solid(name: str, prim, M: np.ndarray, mesh_dir) -> Optional[Solid]:
    """The prim's collision solid; *M* maps its local coordinates to the part frame (metres, z up)."""
    V, F, exact = _prim_mesh(prim)
    if len(V) == 0 or len(F) == 0:
        return None
    T, P = _split(M)
    s = _uniform(P)
    if exact is not None and s is not None:
        kind, p = exact
        if kind == "sphere":
            return Solid(name, T, Shape("sphere", radius=p["radius"] * s))
        return Solid(name, T @ se3.make_T(p["R"], None), Shape("cylinder", radius=p["radius"] * s, length=p["length"] * s))
    V = V @ P.T
    if np.linalg.det(P) < 0:
        F = F[:, ::-1]
    b = box_from_vertices(np.unique(V, axis=0))         # a cube's corners may repeat per face
    if b is not None:
        return Solid(name, T @ b[0], Shape.box(b[1]))
    return Solid(name, T, Shape.mesh(write_obj(V, F, mesh_dir)))


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def _stage_frame(stage) -> np.ndarray:
    """The stage's coordinates -> metres, z up."""
    from pxr import UsdGeom
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    C = np.diag([mpu, mpu, mpu, 1.0])
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        C = se3.make_T(se3.rot_x(math.pi / 2)) @ C          # +Y up -> +Z up
    return C


def _matches(path: str, globs) -> bool:
    p = path
    while p:
        if any(fnmatch.fnmatchcase(p, g) for g in globs):
            return True
        p = p.rsplit("/", 1)[0]
    return False


def load(path, prim: Optional[str] = None, collision=None, frames: Optional[dict] = None,
         mesh_dir=None) -> UsdModel:
    """A USD file as solids in *prim*'s frame (module docstring)."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as e:                       # pragma: no cover - depends on the install
        raise UsdError("reading USD needs pxr: pip install usd-core (Isaac Sim brings its own)") from e
    path = Path(path)
    if not path.exists():
        raise UsdError(f"no such USD file: {path}")
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise UsdError(f"could not open {path}")
    root = stage.GetPseudoRoot() if prim in (None, "", "/") else stage.GetPrimAtPath(prim)
    if not root or not root.IsValid():
        top = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        raise UsdError(f"{path.name}: no prim {prim!r} (top-level prims: {top})")
    C = _stage_frame(stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    W_root = np.eye(4) if root.IsPseudoRoot() else _gf(cache.GetLocalToWorldTransform(root))
    to_part = C @ np.linalg.inv(W_root)            # stage world -> part frame (metres, z up)
    rng = Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    prims = list(rng)
    rel = (lambda p: str(p.GetPath())[1:]) if root.IsPseudoRoot() else \
        (lambda p: str(p.GetPath().MakeRelativePath(root.GetPath())))

    geo = []
    for p in prims:
        if p.GetTypeName() not in GPRIMS:
            continue
        purpose = UsdGeom.Imageable(p).ComputePurpose()
        if purpose == UsdGeom.Tokens.guide:
            continue
        geo.append((p, purpose))

    def enabled(p):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            return False
        a = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
        return a.Get() is not False

    if collision:
        globs = [collision] if isinstance(collision, str) else list(collision)
        chosen = [p for p, _ in geo if _matches(str(p.GetPath()), globs)]
        rule = f"collision globs {globs}"
        if not chosen:
            raise UsdError(f"{path.name}: collision {globs} selects no geometric prim")
    elif any(enabled(p) for p, _ in geo):
        chosen = [p for p, _ in geo if enabled(p)]
        rule = "UsdPhysics.CollisionAPI"
    else:
        proxy = any(pu == UsdGeom.Tokens.proxy for _, pu in geo)
        chosen = [p for p, pu in geo if not (proxy and pu == UsdGeom.Tokens.render)]
        rule = "all geometry" + (" (proxy purpose, render left out)" if proxy else "")

    solids = []
    for p in chosen:
        # local coordinates (stage units) -> the part frame (metres, z up)
        s = _solid(rel(p), p, to_part @ _gf(cache.GetLocalToWorldTransform(p)), mesh_dir)
        if s is not None:
            solids.append(s)
    if not solids:
        raise UsdError(f"{path.name}: no collision geometry under {root.GetPath()} ({rule})")

    kpu = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    mass = None
    if root.HasAPI(UsdPhysics.MassAPI) and (UsdPhysics.MassAPI(root).GetMassAttr().Get() or 0.0) > 0:
        mass = float(UsdPhysics.MassAPI(root).GetMassAttr().Get()) * kpu
    else:
        ms = [float(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0) for p in prims
              if p.HasAPI(UsdPhysics.MassAPI)]
        mass = sum(ms) * kpu if sum(ms) > 0 else None

    out_frames = {}
    if frames:
        unknown = set(frames) - {"prims", "kind"}
        if unknown or "prims" not in frames:
            raise UsdError(f"frames: needs 'prims' (a glob), optional 'kind'; unknown {sorted(unknown)}")
        for p in prims:
            if p == root or not fnmatch.fnmatchcase(str(p.GetPath()), frames["prims"]):
                continue
            Tf, _ = _split(to_part @ _gf(cache.GetLocalToWorldTransform(p)))
            name = p.GetName()
            if name in out_frames:
                raise UsdError(f"{path.name}: two frames named {name!r} under {frames['prims']}")
            out_frames[name] = (Tf, dict(p.GetCustomDataByKey("rsf") or {}))
        if not out_frames:
            raise UsdError(f"{path.name}: frames {frames['prims']!r} matches no prim")
    return UsdModel(path, solids, mass, out_frames, float(UsdGeom.GetStageMetersPerUnit(stage)),
                    str(UsdGeom.GetStageUpAxis(stage)), rule, _visual(path, stage, root, C))


def _visual(path: Path, stage, root, C) -> dict:
    """The looks of the part at *root*: what a stage references to show it.

    ``prims``: the prims to reference (``prim`` itself, or every top-level prim of the
    stage when the part is the whole stage); ``keep_ops``: whether their own transforms
    stay (a whole stage: yes, its prims sit where the stage puts them; ``prim``: no, its
    frame is the part's); ``to_part``: stage coordinates -> the part frame (metres, z up)."""
    if root.IsPseudoRoot():
        prims, keep = [str(p.GetPath()) for p in root.GetChildren() if p.IsDefined() and not p.IsAbstract()], True
    else:
        prims, keep = [str(root.GetPath())], False
    return {"usd": str(Path(path).resolve()), "prims": prims, "keep_ops": keep, "to_part": np.asarray(C).tolist()}


def visual_source(path, prim: Optional[str] = None) -> dict:
    """:func:`_visual` of a USD file used only for its looks (``visual: {model, prim}``)."""
    try:
        from pxr import Usd
    except ImportError as e:                       # pragma: no cover
        raise UsdError("reading USD needs pxr: pip install usd-core (Isaac Sim brings its own)") from e
    path = Path(path)
    if not path.exists():
        raise UsdError(f"no such USD file: {path}")
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    root = stage.GetPseudoRoot() if prim in (None, "", "/") else stage.GetPrimAtPath(prim)
    if not root or not root.IsValid():
        top = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        raise UsdError(f"{path.name}: no prim {prim!r} (top-level prims: {top})")
    return _visual(path, stage, root, _stage_frame(stage))

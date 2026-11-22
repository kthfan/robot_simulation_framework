"""A :class:`~rsf.model.world.World` as a USD stage with physics (UsdPhysics), for Isaac Sim (P5).

Same cell, same physical choices as the MuJoCo compiler (:mod:`rsf.compile.mjcf`
plus :class:`rsf.sim.mujoco_cell.MujocoCell`), written with the core USD schemas
(``usd-core`` is enough to author and test it; Isaac Sim's PhysX reads it):

* ``/World/static/<solid>``: every static solid, a collider without a body;
* ``/World/objects/<object>``: objects as rigid bodies, their solids as
  colliders, mass properties computed here (the object's mass shared by volume,
  as in MuJoCo);
* ``/World/robot``: the URDF as an articulation, written here rather than by
  Isaac's URDF importer so the joints and frames are the ones the planner uses.
  Links joined by fixed joints are one rigid body (``/World/robot/<body>``,
  named after its top link); every link is an ``Xform`` under its body
  (``/World/robot/<body>/<link>``) with its collision meshes (convex hulls, as
  MuJoCo collides them). Revolute and prismatic joints with position drives
  (the arm's PD gains of :class:`~rsf.sim.mujoco_cell.SimParams`; PhysX
  takes angular gains per degree), limits from the URDF; robot links ignore
  gravity (the controller is gravity compensated) and report contacts;
  self-collision off (the planner's job).
* physics materials per class (``robot``, ``object``, ``static``) with
  ``max`` friction combining, as in MuJoCo.

Every prim that stands for a model element carries ``customData``
``rsf:name`` (the solid, object, link or joint name), ``rsf:kind`` and, for
robot links, ``rsf:body``: names in USD are sanitised (``/`` is not allowed).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from rsf.core import se3
from rsf.model.urdf import UrdfModel

ROOT = "/World"
_DEG = 180.0 / math.pi


def prim_name(name: str) -> str:
    """A valid USD prim name for *name* (``tray/incoming/L0/C0`` -> ``tray_incoming_L0_C0``)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return s if s and not s[0].isdigit() else "_" + s


# ── mass properties ──────────────────────────────────────────────────────────
def _shape_inertia(shape, mass: float) -> np.ndarray:
    """Inertia about the shape's centre, in its frame."""
    if shape.kind == "box":
        x, y, z = np.asarray(shape.size, float)
        return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    if shape.kind == "cylinder":
        r, h = shape.radius, shape.length
        a = mass * (3 * r * r + h * h) / 12.0
        return np.diag([a, a, mass * r * r / 2.0])
    if shape.kind == "sphere":
        return np.eye(3) * 0.4 * mass * shape.radius ** 2
    raise ValueError(f"no inertia for a {shape.kind}")


def _volume(shape) -> float:
    if shape.kind == "box":
        return float(np.prod(shape.size))
    if shape.kind == "cylinder":
        return float(math.pi * shape.radius ** 2 * shape.length)
    if shape.kind == "sphere":
        return float(4.0 / 3.0 * math.pi * shape.radius ** 3)
    return 0.0


def combine(parts) -> tuple:
    """``(mass, com, I)`` of rigid parts ``(mass, T_frame, I_about_com_in_frame)``
    given in one frame (``T_frame``: the part's inertial frame); I about the
    combined centre of mass, in that frame."""
    parts = [p for p in parts if p[0] > 0]
    m = sum(p[0] for p in parts)
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(p[0] * np.asarray(p[1])[:3, 3] for p in parts) / m
    I = np.zeros((3, 3))
    for mi, T, Ii in parts:
        R, d = np.asarray(T)[:3, :3], np.asarray(T)[:3, 3] - com
        I += R @ Ii @ R.T + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return m, com, I


def principal(I) -> tuple:
    """``(diagonal, R)`` with ``I = R diag R^T`` (R a proper rotation)."""
    w, V = np.linalg.eigh((np.asarray(I) + np.asarray(I).T) / 2.0)
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    return np.maximum(w, 1e-9), V


def object_mass(obj) -> tuple:
    """``(mass, com, I)`` of an object in its own frame (mass shared by volume)."""
    vols = [_volume(s.shape) for s in obj.solids]
    total = sum(vols) or 1.0
    mass = obj.mass if obj.mass is not None else 1.0
    return combine([(mass * v / total, s.T, _shape_inertia(s.shape, mass * v / total))
                    for s, v in zip(obj.solids, vols) if v > 0])


# ── robot: links grouped into rigid bodies ───────────────────────────────────
def robot_bodies(model: UrdfModel) -> dict:
    """``{link: (body, T_body_link)}``: links joined by fixed joints share the body
    of their top link."""
    out = {model.root: (model.root, np.eye(4))}

    def walk(link):
        for j in model.children.get(link, []):
            if j.type == "fixed":
                body, T = out[link]
                out[j.child] = (body, T @ j.origin)
            else:
                out[j.child] = (j.child, np.eye(4))
            walk(j.child)

    walk(model.root)
    return out


def _x_to(axis) -> np.ndarray:
    """A rotation taking +x to *axis* (PhysX joints turn or slide along their x)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    v, c = np.cross(x, a), float(x @ a)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else se3.rot_z(math.pi)
    return se3.axis_angle(v / np.linalg.norm(v), math.atan2(np.linalg.norm(v), c))


# ── stage ────────────────────────────────────────────────────────────────────
def _gf(T):
    from pxr import Gf
    T = np.asarray(T, float)
    q = se3.quat_wxyz(T[:3, :3])
    return Gf.Vec3d(*T[:3, 3].tolist()), Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:].tolist()))


def _gff(T):
    from pxr import Gf
    p, q = _gf(T)
    return Gf.Vec3f(p), Gf.Quatf(q)


def _xform(prim_xf, T, scale=None):
    from pxr import UsdGeom
    p, q = _gf(T)
    x = UsdGeom.Xformable(prim_xf)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(p)
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(q)
    if scale is not None:
        from pxr import Gf
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(s) for s in scale]))


def _tag(prim, name, kind, **extra):
    prim.SetCustomDataByKey("rsf:name", name)
    prim.SetCustomDataByKey("rsf:kind", kind)
    for k, v in extra.items():
        prim.SetCustomDataByKey(f"rsf:{k}", v)


def _apply(prim, api: str):
    """Apply an API schema by name (PhysX schemas are not in usd-core; Isaac's
    PhysX reads the same tokens)."""
    prim.AddAppliedSchema(api)


def _attr(prim, name, value, typ):
    prim.CreateAttribute(name, typ).Set(value)


def _material(stage, path, friction: float):
    from pxr import Sdf, UsdPhysics, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))     # MuJoCo has one coefficient; same here
    api.CreateRestitutionAttr(0.0)
    _apply(mat.GetPrim(), "PhysxMaterialAPI")
    _attr(mat.GetPrim(), "physxMaterial:frictionCombineMode", "max", Sdf.ValueTypeNames.Token)
    _attr(mat.GetPrim(), "physxMaterial:restitutionCombineMode", "min", Sdf.ValueTypeNames.Token)
    return mat


def _bind(prim, mat):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _collider(prim, p):
    from pxr import Sdf, UsdPhysics
    UsdPhysics.CollisionAPI.Apply(prim)
    _apply(prim, "PhysxCollisionAPI")
    _attr(prim, "physxCollision:contactOffset", float(p.contact_offset), Sdf.ValueTypeNames.Float)
    _attr(prim, "physxCollision:restOffset", 0.0, Sdf.ValueTypeNames.Float)


def _shape_prim(stage, path, shape, T, p, meshes: dict):
    """A collider for *shape* at *T* (in the parent prim's frame)."""
    from pxr import UsdGeom, UsdPhysics, Vt
    if stage.GetPrimAtPath(path):
        raise ValueError(f"{path}: two model elements with the same USD name")
    if shape.kind == "box":
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        _xform(g.GetPrim(), T, scale=np.asarray(shape.size, float))
    elif shape.kind == "sphere":
        g = UsdGeom.Sphere.Define(stage, path)
        g.CreateRadiusAttr(float(shape.radius))
        _xform(g.GetPrim(), T)
    elif shape.kind == "cylinder":
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(float(shape.radius))
        g.CreateHeightAttr(float(shape.length))
        g.CreateAxisAttr("Z")
        _xform(g.GetPrim(), T)
    elif shape.kind == "mesh":
        v, f = _mesh(shape.path, shape.scale, meshes)
        g = UsdGeom.Mesh.Define(stage, path)
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1).astype(np.int32)))
        g.CreateSubdivisionSchemeAttr("none")
        _xform(g.GetPrim(), T)
    else:
        raise ValueError(f"{path}: unsupported shape {shape.kind!r}")
    _collider(g.GetPrim(), p)
    if shape.kind == "mesh":
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("convexHull")
    return g.GetPrim()


def _mesh(path, scale, cache: dict):
    key = (str(path), None if scale is None else tuple(np.asarray(scale, float)))
    if key not in cache:
        import trimesh
        m = trimesh.load(str(path), force="mesh", process=False)
        v = np.asarray(m.vertices, float) * (1.0 if scale is None else np.asarray(scale, float))
        cache[key] = (v, np.asarray(m.faces, np.int64))
    return cache[key]


def _geometry_mass(geos, meshes, density: float = 1000.0) -> list:
    """``(mass, T_com, I)`` of collision geometry at *density* (MuJoCo's default for
    links without ``<inertial>``)."""
    import trimesh
    out = []
    for geo in geos:
        sh = _geometry_shape(geo)
        if sh.kind == "mesh":
            v, f = _mesh(sh.path, sh.scale, meshes)
            tm = trimesh.Trimesh(v, f, process=False)
            mp = tm.mass_properties if tm.is_volume else tm.convex_hull.mass_properties
            mass = float(mp["volume"]) * density
            com = np.asarray(mp["center_mass"], float)
            I = np.asarray(mp["inertia"], float) * density
            out.append((mass, geo.origin @ se3.make_T(None, com), I))
        else:
            mass = _volume(sh) * density
            out.append((mass, geo.origin, _shape_inertia(sh, mass)))
    return out


def _geometry_shape(geo):
    from rsf.model.world import Shape
    if geo.kind == "mesh":
        return Shape.mesh(geo.path, None if np.allclose(geo.scale, 1.0) else geo.scale)
    if geo.kind == "box":
        return Shape.box(geo.size)
    return Shape(geo.kind, radius=geo.radius, length=geo.length)


def _mass_api(prim, mass, com, I):
    from pxr import Gf, UsdPhysics
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(c) for c in com]))
    d, R = principal(I)
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(x) for x in d]))
    q = se3.quat_wxyz(R)
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[0]), *[float(x) for x in q[1:]]))


def world_stage(world, path=None, *, params=None, robot: bool = True, objects: bool = True):
    """The world as a ``Usd.Stage`` (written to *path* if given: ``.usda`` / ``.usd``).

    *params*: :class:`rsf.sim.isaac_cell.IsaacParams` (default values if None)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    if params is None:
        from rsf.sim.isaac_cell import IsaacParams
        params = IsaacParams()
    p = params
    stage = Usd.Stage.CreateNew(str(path)) if path is not None else Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)
    _tag(root, world.name or "rsf_world", "world")

    scene = UsdPhysics.Scene.Define(stage, f"{ROOT}/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    sp = scene.GetPrim()
    _apply(sp, "PhysxSceneAPI")
    _attr(sp, "physxScene:timeStepsPerSecond", int(round(1.0 / p.timestep)), Sdf.ValueTypeNames.UInt)
    _attr(sp, "physxScene:solverType", p.solver, Sdf.ValueTypeNames.Token)
    _attr(sp, "physxScene:enableGPUDynamics", bool(p.gpu_dynamics), Sdf.ValueTypeNames.Bool)
    _attr(sp, "physxScene:broadphaseType", "GPU" if p.gpu_dynamics else "MBP", Sdf.ValueTypeNames.Token)
    _attr(sp, "physxScene:enableCCD", False, Sdf.ValueTypeNames.Bool)
    _attr(sp, "physxScene:frictionType", "patch", Sdf.ValueTypeNames.Token)

    fr = p.friction
    mats = {k: _material(stage, f"{ROOT}/materials/{k}", fr[k]) for k in ("robot", "object", "static")}
    meshes: dict = {}

    UsdGeom.Scope.Define(stage, f"{ROOT}/static")
    for s in world.solids(objects=False):
        prim = _shape_prim(stage, f"{ROOT}/static/{prim_name(s.name)}", s.shape, s.T, p, meshes)
        _tag(prim, s.name, "static", entity=s.entity)
        _bind(prim, mats["static"])

    if objects:
        UsdGeom.Scope.Define(stage, f"{ROOT}/objects")
        for o in world.objects:
            bp = f"{ROOT}/objects/{prim_name(o.name)}"
            b = UsdGeom.Xform.Define(stage, bp).GetPrim()
            _xform(b, o.T)
            _tag(b, o.name, "object")
            UsdPhysics.RigidBodyAPI.Apply(b)
            _mass_api(b, *object_mass(o))
            for s in o.solids:
                c = _shape_prim(stage, f"{bp}/{prim_name(s.name)}", s.shape, s.T, p, meshes)
                _tag(c, f"{o.name}/{s.name}", "object_solid", object=o.name)
                _bind(c, mats["object"])

    if robot and world.robot is not None:
        _robot(stage, world.robot, p, mats["robot"], meshes)
    if path is not None:
        stage.GetRootLayer().Save()
    return stage


def _robot(stage, spec, p, mat, meshes):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    model = spec.urdf if isinstance(spec.urdf, UrdfModel) else UrdfModel.from_file(spec.urdf)
    rp = f"{ROOT}/robot"
    rprim = UsdGeom.Xform.Define(stage, rp).GetPrim()
    _xform(rprim, spec.base)
    _tag(rprim, model.name, "robot", tool_frame=spec.tool_frame)
    UsdPhysics.ArticulationRootAPI.Apply(rprim)
    _apply(rprim, "PhysxArticulationAPI")
    _attr(rprim, "physxArticulation:enabledSelfCollisions", False, Sdf.ValueTypeNames.Bool)
    _attr(rprim, "physxArticulation:solverPositionIterationCount", int(p.position_iterations), Sdf.ValueTypeNames.Int)
    _attr(rprim, "physxArticulation:solverVelocityIterationCount", int(p.velocity_iterations), Sdf.ValueTypeNames.Int)
    groups = robot_bodies(model)
    fk = model.fk()                                     # q = 0, in the robot root frame
    bodies = sorted({b for b, _ in groups.values()}, key=list(model.links).index)
    for b in bodies:
        bprim = UsdGeom.Xform.Define(stage, f"{rp}/{prim_name(b)}").GetPrim()
        _xform(bprim, fk[b])
        _tag(bprim, b, "robot_body")
        UsdPhysics.RigidBodyAPI.Apply(bprim)
        _apply(bprim, "PhysxRigidBodyAPI")
        _attr(bprim, "physxRigidBody:disableGravity", True, Sdf.ValueTypeNames.Bool)   # gravity compensated
        _apply(bprim, "PhysxContactReportAPI")
        _attr(bprim, "physxContactReport:threshold", 0.0, Sdf.ValueTypeNames.Float)
        parts = []
        for link, (bb, T_bl) in groups.items():
            if bb != b:
                continue
            L = model.links[link]
            lp = f"{rp}/{prim_name(b)}/{prim_name(link)}" if link != b else f"{rp}/{prim_name(b)}/{prim_name(b)}_frame"
            lprim = UsdGeom.Xform.Define(stage, lp).GetPrim()
            _xform(lprim, T_bl)
            _tag(lprim, link, "robot_link", body=b)
            for k, geo in enumerate(L.collisions):
                c = _shape_prim(stage, f"{lp}/collision_{k}", _geometry_shape(geo), geo.origin, p, meshes)
                _tag(c, link, "robot_collision", body=b)
                _bind(c, mat)
            if L.mass:
                I = L.inertia if L.inertia is not None else np.eye(3) * 1e-6
                parts.append((L.mass, T_bl @ L.com, I))
            else:                                       # as MuJoCo infers it: the geometry at 1000 kg/m^3
                parts.extend((mi, T_bl @ T, I) for mi, T, I in _geometry_mass(L.collisions, meshes))
        m, com, I = combine(parts)
        if m <= 0:                                      # PhysX needs mass on every link
            m, com, I = p.min_link_mass, np.zeros(3), np.eye(3) * p.min_link_mass * 1e-4
        _mass_api(bprim, m, com, I)

    # the root body fixed to the world
    fj = UsdPhysics.FixedJoint.Define(stage, f"{rp}/joints/root_fixed")
    fj.CreateBody1Rel().SetTargets([Sdf.Path(f"{rp}/{prim_name(model.root)}")])
    p0, q0 = _gff(np.asarray(spec.base, float))
    fj.CreateLocalPos0Attr(p0)
    fj.CreateLocalRot0Attr(q0)
    fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    _tag(fj.GetPrim(), "root_fixed", "robot_joint")

    arm = 0
    for jn in _tree_joints(model):
        j = model.joints[jn]
        if not j.movable:
            continue
        b0, T_b0_parent = groups[j.parent]
        R_a = _x_to(j.axis)
        T0 = T_b0_parent @ j.origin @ se3.make_T(R_a)
        T1 = se3.make_T(R_a)                            # the child link is its body's top link
        jp = f"{rp}/joints/{prim_name(jn)}"
        if j.type == "prismatic":
            J = UsdPhysics.PrismaticJoint.Define(stage, jp)
            lo, hi, drive = j.lower, j.upper, "linear"
        else:
            J = UsdPhysics.RevoluteJoint.Define(stage, jp)
            lo, hi, drive = j.lower * _DEG, j.upper * _DEG, "angular"
        J.CreateAxisAttr("X")
        J.CreateBody0Rel().SetTargets([Sdf.Path(f"{rp}/{prim_name(b0)}")])
        J.CreateBody1Rel().SetTargets([Sdf.Path(f"{rp}/{prim_name(j.child)}")])
        for i, T in ((0, T0), (1, T1)):
            pp, qq = _gff(T)
            getattr(J, f"CreateLocalPos{i}Attr")(pp)
            getattr(J, f"CreateLocalRot{i}Attr")(qq)
        if j.type != "continuous" and np.isfinite(lo) and np.isfinite(hi):
            J.CreateLowerLimitAttr(float(lo))
            J.CreateUpperLimitAttr(float(hi))
        d = UsdPhysics.DriveAPI.Apply(J.GetPrim(), drive)
        d.CreateTypeAttr("force")
        jprim = J.GetPrim()
        _apply(jprim, "PhysxJointAPI")
        if drive == "angular":
            kp, kv = p.arm_kp[arm], p.arm_kv[arm]
            arm += 1
            d.CreateStiffnessAttr(float(kp / _DEG))       # per degree
            d.CreateDampingAttr(float(kv / _DEG))
            d.CreateMaxForceAttr(float(p.arm_max_torque))
            _attr(jprim, "physxJoint:armature", float(p.armature), Sdf.ValueTypeNames.Float)
        else:
            d.CreateStiffnessAttr(float(p.jaw_kp))
            d.CreateDampingAttr(float(2.0 * math.sqrt(p.jaw_kp * p.jaw_armature)))
            d.CreateMaxForceAttr(float(p.jaw_force))
            _attr(jprim, "physxJoint:armature", float(p.jaw_armature), Sdf.ValueTypeNames.Float)
        d.CreateTargetPositionAttr(0.0)
        _attr(jprim, "physxJoint:maxJointVelocity", float(p.max_joint_velocity), Sdf.ValueTypeNames.Float)
        _tag(jprim, jn, "robot_joint", type=j.type)


def _tree_joints(model) -> list:
    out = []

    def walk(link):
        for j in model.children.get(link, []):
            out.append(j.name)
            walk(j.child)

    walk(model.root)
    return out


# ── reading a stage back (tests, and the Isaac cell's name maps) ──────────────
def _T_of(pos, rot) -> np.ndarray:
    q = np.array([rot.GetReal(), *rot.GetImaginary()], float)
    return se3.make_T(se3.quat_to_matrix(q), np.array(pos, float))


def stage_fk(stage, q: dict) -> dict:
    """World pose of every robot body for joint values *q* (rad / m), composed from the
    stage's own joints: ``T_b1 = T_b0 · local0 · motion · local1⁻¹``. Tests use it to
    check the authored articulation against the URDF."""
    from pxr import UsdGeom, UsdPhysics
    joints = [p for p in stage.Traverse() if p.GetCustomDataByKey("rsf:kind") == "robot_joint"]
    poses = {}
    root = next(p for p in joints if p.GetCustomDataByKey("rsf:name") == "root_fixed")
    fj = UsdPhysics.FixedJoint(root)
    poses[str(fj.GetBody1Rel().GetTargets()[0])] = _T_of(fj.GetLocalPos0Attr().Get(), fj.GetLocalRot0Attr().Get())
    pending = [p for p in joints if p != root]
    while pending:
        rest = []
        for p in pending:
            J = UsdPhysics.Joint(p)
            b0, b1 = str(J.GetBody0Rel().GetTargets()[0]), str(J.GetBody1Rel().GetTargets()[0])
            if b0 not in poses:
                rest.append(p)
                continue
            L0 = _T_of(J.GetLocalPos0Attr().Get(), J.GetLocalRot0Attr().Get())
            L1 = _T_of(J.GetLocalPos1Attr().Get(), J.GetLocalRot1Attr().Get())
            v = float(q.get(p.GetCustomDataByKey("rsf:name"), 0.0))
            if p.IsA(UsdPhysics.PrismaticJoint):
                M = se3.make_T(None, (v, 0.0, 0.0))
            else:
                M = se3.make_T(se3.rot_x(v))
            poses[b1] = poses[b0] @ L0 @ M @ se3.inv(L1)
        if len(rest) == len(pending):
            raise ValueError("joints not connected to the root")
        pending = rest
    return {stage.GetPrimAtPath(k).GetCustomDataByKey("rsf:name"): v for k, v in poses.items()}


def link_map(stage) -> dict:
    """``{link: (body, T_body_link)}`` as authored."""
    from pxr import UsdGeom
    out = {}
    for p in stage.Traverse():
        if p.GetCustomDataByKey("rsf:kind") == "robot_link":
            M = UsdGeom.Xformable(p).GetLocalTransformation()
            out[p.GetCustomDataByKey("rsf:name")] = (p.GetCustomDataByKey("rsf:body"), np.array(M).T)
    return out

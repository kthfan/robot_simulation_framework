"""A :class:`~rsf.model.world.World` as a MuJoCo model (``mujoco.MjSpec``).

* static solids become geoms of the world body (boxes stay boxes; meshes are
  loaded as MuJoCo meshes, which MuJoCo collides as their convex hulls);
* objects become free bodies whose geoms share the object's mass by volume;
* the robot is MuJoCo's own URDF import, attached at its base pose under the
  ``robot/`` prefix, with a position actuator per movable joint.

MuJoCo is the CPU simulator (P5); this compiler is shared with it so the
simulated cell is the planned one.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from rsf.core import se3
from rsf.model.urdf import UrdfModel

ROBOT_PREFIX = "robot/"


def _pose(T):
    T = np.asarray(T, float)
    return T[:3, 3].tolist(), se3.quat_wxyz(T[:3, :3]).tolist()


def _volume(shape) -> float:
    if shape.kind == "box":
        return float(np.prod(shape.size))
    if shape.kind == "cylinder":
        return float(np.pi * shape.radius ** 2 * shape.length)
    if shape.kind == "sphere":
        return float(4.0 / 3.0 * np.pi * shape.radius ** 3)
    return 0.0


def _add_geom(spec, body, solid, T, *, mass: Optional[float] = None, mesh_names: dict, name: Optional[str] = None):
    import mujoco
    pos, quat = _pose(T)
    g = body.add_geom(name=name or solid.name, pos=pos, quat=quat)
    sh = solid.shape
    if sh.kind == "box":
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = (np.asarray(sh.size, float) / 2.0).tolist()
    elif sh.kind == "cylinder":
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [sh.radius, sh.length / 2.0, 0.0]
    elif sh.kind == "sphere":
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [sh.radius, 0.0, 0.0]
    elif sh.kind == "mesh":
        key = (str(sh.path), None if sh.scale is None else tuple(sh.scale))
        if key not in mesh_names:
            name = f"mesh{len(mesh_names)}"
            m = spec.add_mesh(name=name, file=str(sh.path))
            if sh.scale is not None:
                m.scale = list(sh.scale)
            mesh_names[key] = name
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh_names[key]
    else:
        raise ValueError(f"{solid.name}: unsupported shape {sh.kind!r}")
    if mass is not None:
        g.mass = mass
    return g


def robot_spec(urdf, jaw_equal: bool = False):
    """MuJoCo's import of *urdf* (absolute mesh paths, fixed links kept as bodies)."""
    import mujoco
    model = urdf if isinstance(urdf, UrdfModel) else UrdfModel.from_file(urdf)
    root = ET.fromstring(model.to_xml())                 # absolute mesh paths
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", strippath="false", fusestatic="false", discardvisual="true",
                  balanceinertia="true")
    root.remove(mj)
    root.insert(0, mj)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def world_spec(world, *, robot: bool = True, objects: bool = True, kp: float = 2000.0, timestep: float = 0.002):
    """The world as an ``MjSpec`` (call ``.compile()`` for an ``MjModel``)."""
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = world.name or "rsf_world"
    spec.option.timestep = timestep
    meshes: dict = {}
    wb = spec.worldbody
    for s in world.solids(objects=False):
        _add_geom(spec, wb, s, s.T, mesh_names=meshes)
    if objects:
        for o in world.objects:
            pos, quat = _pose(o.T)
            b = wb.add_body(name=o.name, pos=pos, quat=quat)
            b.add_freejoint(name=f"{o.name}/free")
            vols = [_volume(s.shape) for s in o.solids]
            total = sum(vols) or 1.0
            for s, v in zip(o.solids, vols):
                _add_geom(spec, b, s, s.T, mesh_names=meshes, name=f"{o.name}/{s.name}",
                          mass=None if o.mass is None else o.mass * v / total)
    if robot and world.robot is not None:
        rs = robot_spec(world.robot.urdf)
        pos, quat = _pose(world.robot.base)
        frame = wb.add_frame(pos=pos, quat=quat)
        spec.attach(rs, frame=frame, prefix=ROBOT_PREFIX)
        for j in spec.joints:
            if j.name.startswith(ROBOT_PREFIX) and j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                a = spec.add_actuator(name=f"{j.name}/pos", target=j.name)
                a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                a.set_to_position(kp=kp, kv=2.0 * np.sqrt(kp))
    return spec

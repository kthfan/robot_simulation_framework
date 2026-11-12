"""Gripper geometry the TM sequence needs, read from the robot's URDF.

Read, not transcribed (AI-Robotics-V2 D22): change the gripper asset and the
sequence follows. The layout names the geometry in ``process.push_plate``::

    process:
      push_plate: {geometry: air_cylinder_plate, margin: 0.002}

A robot without that geometry (``tm5s_egu50_no_air_cylinder.urdf``) has no push
plate, and the sequence is the D28 one.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from rsf.core import se3
from rsf.model.urdf import UrdfModel
from rsf.model.world import solids_from_urdf

TOOL_FRAME = "flange"


@lru_cache(maxsize=16)
def _extent(urdf: str, geometry: str, frame: str):
    """``(lo, hi)`` of a named collision geometry in *frame* at q = 0, or None."""
    m = UrdfModel.from_file(urdf)
    inv = se3.inv(m.fk()[frame])
    for s in solids_from_urdf(m):
        if s.name.split("/")[-1] != geometry:
            continue
        if s.shape.kind == "box":
            P = s.corners()
        else:
            import trimesh
            V = np.asarray(trimesh.load(s.shape.path, force="mesh").vertices)
            P = (s.T @ np.c_[V, np.ones(len(V))].T).T[:, :3]
        P = (inv @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        return P.min(0), P.max(0)
    return None


def tool_geometry(world) -> dict:
    """``{"push_front_z": ...}`` (flange z of the push plate's tray-side face), or
    ``{}`` when the layout names no push plate or the robot has none."""
    spec = world.process.get("push_plate") or {}
    if not spec.get("geometry") or world.robot is None:
        return {}
    ext = _extent(str(Path(world.robot.urdf).resolve()), spec["geometry"], TOOL_FRAME)
    if ext is None:
        return {}
    return {"push_front_z": float(ext[1][2]), "push_plate_extent": (ext[0].tolist(), ext[1].tolist())}

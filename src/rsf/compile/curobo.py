"""A :class:`~rsf.model.world.World` as a cuRobo scene dictionary.

Poses are in the robot base frame (cuRobo plans in the base frame), as
``[x, y, z, qw, qx, qy, qz]``. Box solids become ``cuboid``, mesh solids
``mesh``, cylinders ``cylinder``. Spheres are not supported by cuRobo scenes and
raise.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from rsf.core import se3


def _pose(T) -> list:
    return [float(v) for v in T[:3, 3]] + [float(v) for v in se3.quat_wxyz(T[:3, :3])]


def scene_dict(solids: Iterable, base, *, exclude: Optional[Callable] = None) -> dict:
    """``{"cuboid": {...}, "mesh": {...}, "cylinder": {...}}`` with poses relative to *base*."""
    inv = se3.inv(np.asarray(base, dtype=float))
    out: dict = {}
    for s in solids:
        if exclude is not None and exclude(s):
            continue
        T = inv @ s.T
        kind = s.shape.kind
        if kind == "box":
            entry = ("cuboid", {"dims": [float(v) for v in s.shape.size], "pose": _pose(T)})
        elif kind == "mesh":
            e = {"file_path": str(s.shape.path), "pose": _pose(T)}
            if s.shape.scale is not None:
                e["scale"] = [float(v) for v in s.shape.scale]
            entry = ("mesh", e)
        elif kind == "cylinder":
            entry = ("cylinder", {"radius": float(s.shape.radius), "height": float(s.shape.length), "pose": _pose(T)})
        else:
            raise ValueError(f"{s.name}: cuRobo scenes have no {kind!r} obstacles")
        table = out.setdefault(entry[0], {})
        if s.name in table:
            raise ValueError(f"duplicate obstacle name {s.name!r}")
        table[s.name] = entry[1]
    return out


def world_scene(world, *, objects: bool = True, exclude_objects=(), exclude=None) -> dict:
    """The world's obstacles for its robot."""
    if world.robot is None:
        raise ValueError("the world has no robot")
    return scene_dict(world.solids(objects=objects, exclude_objects=exclude_objects), world.robot.base,
                      exclude=exclude)

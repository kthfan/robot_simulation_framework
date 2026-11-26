"""The gripper and the block as the skills need them, read from the URDFs.

Read, not copied: change ``tools/make_assets.py`` (or swap the URDFs) and the
skills follow.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from rsf.core import se3
from rsf.model.urdf import UrdfModel
from rsf.model.world import solids_from_urdf

FINGERS = ("finger_right_joint", "finger_left_joint")
FINGER_LINKS = ("finger_right", "finger_left")


@lru_cache(maxsize=8)
def _gripper(urdf: str, tool_frame: str) -> dict:
    m = UrdfModel.from_file(urdf)
    fk = m.fk()
    T_tool = fk[tool_frame]
    inv = se3.inv(T_tool)
    lo, hi = m.limits(list(FINGERS))
    tips, body_far = [], None
    for s in solids_from_urdf(m):
        link = s.name.split("/")[0]
        P = s.corners() if s.shape.kind == "box" else None
        if P is None:
            continue
        z = (inv @ np.c_[P, np.ones(8)].T)[2]            # along the approach, tool frame
        if link in FINGER_LINKS:
            tips.append(z.max())
        elif link == "gripper_body":
            body_far = z.max()
    return {"finger_tip_z": float(max(tips)), "body_far_z": float(body_far),
            "finger_lower": float(lo.max()), "finger_upper": float(hi.min())}


def gripper(world) -> dict:
    """``finger_tip_z`` / ``body_far_z``: the finger tips and the body's far face along
    the tool's approach axis (tool frame z; negative: behind the tcp); finger limits."""
    return _gripper(str(Path(world.robot.urdf).resolve()), world.robot.tool_frame)


def block_size(world, key: str) -> np.ndarray:
    """The block's extent (x, y, z) from its solids (template at feature *key*)."""
    obj = world.object_at(key)
    P = np.vstack([s.corners() for s in obj.world_solids()])
    Rinv = np.asarray(obj.T)[:3, :3].T
    L = (P - np.asarray(obj.T)[:3, 3]) @ Rinv.T
    return L.max(0) - L.min(0)

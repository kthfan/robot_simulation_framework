"""Kinematics backends: FK, exact IK branches, and the robot's place in the world.

``KinematicsBackend`` is what planners use; :class:`ArmKinematics` adds the
robot base pose so callers work in world coordinates. Backends:

* :class:`rsf.kinematics.eaik.EaikKinematics` — analytic IK (EAIK fork) with the
  numerical safety layer of robot_arm_motion_planner's ``RobustIK``. CPU.
* cuRobo (GPU, P7) will implement the same protocol.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from rsf.core import se3


@runtime_checkable
class KinematicsBackend(Protocol):
    joint_names: list
    lower: np.ndarray
    upper: np.ndarray

    def fk(self, q) -> np.ndarray:
        """Tip pose in the base frame."""

    def ik(self, T, *, within_limits: bool = True, expand_turns: bool = True) -> list:
        """Every exact solution for tip pose *T* (base frame)."""


class ArmKinematics:
    """A backend placed in the world at ``base`` (world pose of the base link)."""

    def __init__(self, backend: KinematicsBackend, base):
        self.backend = backend
        self.base = np.asarray(base, dtype=float)
        self._inv = se3.inv(self.base)

    @property
    def joint_names(self) -> list:
        return self.backend.joint_names

    @property
    def lower(self) -> np.ndarray:
        return self.backend.lower

    @property
    def upper(self) -> np.ndarray:
        return self.backend.upper

    def fk(self, q) -> np.ndarray:
        return self.base @ self.backend.fk(q)

    def ik(self, T_world, **kw) -> list:
        return self.backend.ik(self._inv @ np.asarray(T_world, dtype=float), **kw)

    def refine(self, T_world, q0, *, tol_pos: float = 1e-7, tol_rot: float = 1e-6):
        """Local IK from *q0* (same branch), or None if the backend cannot or it misses."""
        if not hasattr(self.backend, "refine"):
            return None
        q, dp, dr = self.backend.refine(self._inv @ np.asarray(T_world, dtype=float), q0)
        return q if dp <= tol_pos and dr <= tol_rot else None

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian in the world frame (6 x n)."""
        J = self.backend.jacobian(q)
        R = self.base[:3, :3]
        return np.vstack([R @ J[:3], R @ J[3:]])

    def manipulability(self, q) -> float:
        """Yoshikawa's measure sqrt(det(J J^T)); 0 at a singularity."""
        J = self.jacobian(q)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def nearest(self, T_world, q_ref: Sequence[float]):
        """The solution nearest *q_ref* (largest joint difference), or None."""
        qs = self.ik(T_world)
        if not qs:
            return None
        q_ref = np.asarray(q_ref, dtype=float)
        return min(qs, key=lambda q: (float(np.abs(q - q_ref).max()), float(np.abs(q - q_ref).sum())))


def load_arm(world, backend: str = "eaik", **kw) -> ArmKinematics:
    """The world's robot as an :class:`ArmKinematics`, chain root -> tool frame."""
    if world.robot is None:
        raise ValueError("the world has no robot")
    if backend == "eaik":
        from rsf.kinematics.eaik import EaikKinematics
        be = EaikKinematics(world.robot.urdf, world.robot.tool_frame, **kw)
    else:
        from rsf.model import registry
        be = registry.get("kinematics", backend)(world, **kw)
    return ArmKinematics(be, world.robot.base)

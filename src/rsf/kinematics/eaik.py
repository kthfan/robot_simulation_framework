"""Analytic IK with EAIK (kthfan fork, ``1.2.2+robust.1``) and ``RobustIK``.

``RobustIK`` (robot_arm_motion_planner) refines every EAIK candidate by Newton
steps on the original, calibrated model, rejects least-squares fill-ins, adds
twin seeds near singularities, applies joint limits and +-2*pi variants. That is
the fix for AI-Robotics-V2 findings/07-08 (EAIK solutions missing reachable
poses on the calibrated TM5S) and replaces that project's own ``refine`` /
``ik_branches`` / ``_turns``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from rsf.model.urdf import UrdfModel


class EaikKinematics:
    def __init__(self, urdf, tip_link: str, base_link: Optional[str] = None,
                 tol_pos: float = 1e-9, tol_rot: float = 1e-9,
                 parallel_tol: float = 2e-3, intersect_tol: float = 1e-3):
        from eaik.IK_URDF import UrdfRobot
        from motion_planner.robust_eaik import RobustIK

        model = urdf if isinstance(urdf, UrdfModel) else UrdfModel.from_file(urdf)
        chain = model.subchain(tip_link, base_link)
        movable = [j for j in chain.chain(tip_link) if chain.joints[j].movable]
        fd, path = tempfile.mkstemp(suffix=".urdf", prefix="rsf_ik_")
        os.close(fd)
        try:
            Path(path).write_text(chain.to_xml())
            robot = UrdfRobot(path, [], tip_link=tip_link, parallel_tol=parallel_tol, intersect_tol=intersect_tol)
        finally:
            os.unlink(path)
        lo, hi = chain.limits(movable)
        self.joint_names = movable
        self.lower, self.upper = lo, hi
        self.tip_link = tip_link
        self.base_link = chain.root
        self.solver = RobustIK(robot, lo, hi, tol_pos=tol_pos, tol_rot=tol_rot)
        self.model = chain

    def fk(self, q) -> np.ndarray:
        return np.asarray(self.solver.fk(np.asarray(q, dtype=float)), dtype=float)

    def ik(self, T, *, within_limits: bool = True, expand_turns: bool = True) -> list:
        res = self.solver.ik(np.asarray(T, dtype=float), within_limits=within_limits, expand_2pi=expand_turns)
        return [np.asarray(q, dtype=float) for q in res.Q]

    def ik_batch(self, Ts, *, num_threads: int = 4, **kw) -> list:
        res = self.solver.ik_batch([np.asarray(T, dtype=float) for T in Ts], num_threads=num_threads, **kw)
        return [[np.asarray(q, dtype=float) for q in r.Q] for r in res]

    def refine(self, T, q0):
        """Newton refinement from *q0* onto *T* (base frame): ``(q, dp, dr)``. Stays on q0's branch."""
        q, dp, dr = self.solver.refine(np.asarray(T, dtype=float), np.asarray(q0, dtype=float))
        return np.asarray(q, dtype=float), dp, dr

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6 x n: linear; angular) of the tip in the base frame."""
        _, J = self.solver._fk_jac(np.asarray(q, dtype=float))
        return np.asarray(J, dtype=float)

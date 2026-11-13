"""Cartesian motions: follow a sequence of tool poses with one continuous IK branch.

A motion is a list of tool poses (world frame), dense enough that consecutive
joint solutions are close. :func:`walk` takes, at every pose, the exact IK
solution nearest the previous one and refuses a step larger than ``max_step``
or one that changes the arm configuration (:class:`ContinuityRule`), which is
how a singularity or a branch flip shows up. Time parameterisation is P4.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from rsf.planning.chain import ContinuityRule, turns


def nearest_solution(arm, T, q_prev, lo=None, hi=None):
    """The exact solution for *T* nearest *q_prev* over every in-limit turn, or None."""
    q_prev = np.asarray(q_prev, dtype=float)
    lo = arm.lower if lo is None else lo
    hi = arm.upper if hi is None else hi
    best, best_d = None, np.inf
    for q in arm.ik(T, expand_turns=False):
        for qt in turns(q, lo, hi):
            d = float(np.abs(qt - q_prev).max())
            if d < best_d:
                best, best_d = qt, d
    return best


def walk(arm, poses: Sequence, q0, *, max_step: float = 0.2,
         rule: ContinuityRule = ContinuityRule(max_step=np.inf)) -> Optional[list]:
    """Joint configurations along *poses* starting from *q0* (which must reach
    ``poses[0]``), or None when the arm cannot follow them continuously."""
    qs = [np.asarray(q0, dtype=float)]
    for T in poses[1:]:
        q = nearest_solution(arm, T, qs[-1])
        if q is None or float(np.abs(q - qs[-1]).max()) > max_step or not rule.continuous(qs[-1], q):
            return None
        qs.append(q)
    return qs

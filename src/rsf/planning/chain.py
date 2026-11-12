"""Joint configurations that move through a sequence of poses continuously.

Generalised from AI-Robotics-V2 ``check_cell`` (M3). A sequence of phases
(``pre_insert -> insert -> lift``) must be served by one IK branch: no elbow or
wrist flip, no large joint jump. Joints whose range exceeds a turn can reach one
pose a whole turn apart, so every in-limit ``q + 2*pi*k`` form is considered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ContinuityRule:
    """When two consecutive configurations count as one continuous motion.

    *sign_joints*: indices whose sign is the arm configuration (TM5S: the elbow,
    joint 3, and the wrist, joint 5) and may not flip. *max_step*: largest change
    of any joint (rad).
    """
    sign_joints: tuple = (2, 4)
    max_step: float = 1.0

    def continuous(self, a, b) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if any(np.sign(a[j]) != np.sign(b[j]) for j in self.sign_joints):
            return False
        return float(np.max(np.abs(a - b))) <= self.max_step


def turns(q, lo, hi) -> list:
    """Every in-limit way to write *q*, joint by joint a whole turn apart."""
    q = np.asarray(q, dtype=float)
    per_joint = [[v + 2 * math.pi * k for k in range(-2, 3) if lo[j] <= v + 2 * math.pi * k <= hi[j]]
                 for j, v in enumerate(q)]
    return [np.array(c) for c in product(*per_joint)]


def chain_joints(qs: Sequence, lo, hi, rule: ContinuityRule = ContinuityRule()) -> Optional[list]:
    """Joint values that move through *qs* continuously (any in-limit turn), or None."""
    def walk(done):
        if len(done) == len(qs):
            return done
        for q in turns(qs[len(done)], lo, hi):
            if done and not rule.continuous(done[-1], q):
                continue
            got = walk(done + [q])
            if got is not None:
                return got
        return None
    return walk([])


def best_chain(candidates: Sequence[Sequence[dict]], lo, hi, rule: ContinuityRule = ContinuityRule()):
    """Pick one candidate per phase: passing (``ok``), continuous, best worst ``score``.

    *candidates*: per phase, a list of ``{"q", "ok", "score"}``. Returns
    ``(indices, joint values)`` or ``(None, None)``.
    """
    options = [[i for i, c in enumerate(phase) if c["ok"]] for phase in candidates]
    best, best_q, best_score = None, None, -math.inf
    for idx in product(*options):
        picked = [phase[i] for phase, i in zip(candidates, idx)]
        score = min(c["score"] for c in picked)
        if score <= best_score:
            continue
        qs = chain_joints([c["q"] for c in picked], lo, hi, rule)
        if qs is not None:
            best, best_q, best_score = idx, qs, score
    return best, best_q

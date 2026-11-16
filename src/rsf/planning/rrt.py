"""CPU fallback for free moves: RRT-Connect in joint space, then shortcutting.

Used when no GPU is at hand (development, CI). cuRobo stays the planner of
record (``tools/kaggle/curobo_free_plan.py``); either way the result goes
through the same CPU gate (:func:`rsf.planning.plan.check_segment`).

The planner only needs ``valid(q) -> bool`` and the joint box; edges are
checked by sampling every ``step`` rad (max-norm), the gate's resolution.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


def edge_ok(valid: Callable, a: np.ndarray, b: np.ndarray, step: float) -> bool:
    k = max(1, math.ceil(float(np.abs(b - a).max()) / step))
    return valid(b) and all(valid(a + (b - a) * (i / k)) for i in _bisection(k))


def _bisection(k: int):
    """1..k-1, middle first, then the middles of the halves, ... (finds a hit early)."""
    out, spans = [], [(0, k)]
    while spans:
        nxt = []
        for lo, hi in spans:
            if hi - lo < 2:
                continue
            m = (lo + hi) // 2
            out.append(m)
            nxt += [(lo, m), (m, hi)]
        spans = nxt
    return out


class _Tree:
    def __init__(self, q0):
        self.Q = [np.asarray(q0, float)]
        self.parent = [-1]

    def nearest(self, q):
        A = np.asarray(self.Q)
        return int(np.argmin(np.abs(A - q).max(axis=1)))

    def add(self, q, p):
        self.Q.append(q)
        self.parent.append(p)
        return len(self.Q) - 1

    def path_to(self, i):
        out = []
        while i >= 0:
            out.append(self.Q[i])
            i = self.parent[i]
        return out[::-1]


def rrt_connect(valid: Callable, q_start, q_goal, lower, upper, *, step: float = 0.01,
                extend: float = 0.3, max_iter: int = 4000, seed: int = 0,
                stats: Optional[dict] = None) -> Optional[np.ndarray]:
    """A collision-free joint path ``q_start -> q_goal`` (rows), or ``None``."""
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    qs, qg = np.asarray(q_start, float), np.asarray(q_goal, float)
    if edge_ok(valid, qs, qg, step):
        return np.array([qs, qg])
    ta, tb = _Tree(qs), _Tree(qg)
    swapped = False

    def steer(t, q):
        i = t.nearest(q)
        a = t.Q[i]
        d = q - a
        n = float(np.abs(d).max())
        b = q if n <= extend else a + d * (extend / n)
        return i, b

    for it in range(max_iter):
        target = qg if (not swapped and rng.random() < 0.1) else rng.uniform(lower, upper)
        i, b = steer(ta, target)
        if not edge_ok(valid, ta.Q[i], b, step):
            ta, tb, swapped = tb, ta, not swapped
            continue
        ia = ta.add(b, i)
        # connect: greedy extension of the other tree toward b
        while True:
            j, c = steer(tb, b)
            if not edge_ok(valid, tb.Q[j], c, step):
                break
            jb = tb.add(c, j)
            if np.abs(c - b).max() < 1e-12:
                pa, pb = ta.path_to(ia), tb.path_to(jb)
                path = pa + pb[::-1][1:]
                if swapped:
                    path = path[::-1]
                if stats is not None:
                    stats.update(iterations=it + 1, nodes=len(ta.Q) + len(tb.Q))
                return np.array(path)
        ta, tb, swapped = tb, ta, not swapped
    if stats is not None:
        stats.update(iterations=max_iter, nodes=len(ta.Q) + len(tb.Q))
    return None


def shortcut(valid: Callable, path: np.ndarray, *, step: float = 0.01, rounds: int = 60,
             seed: int = 0) -> np.ndarray:
    """Random shortcutting, then dropping redundant via points."""
    rng = np.random.default_rng(seed)
    P = [np.asarray(p, float) for p in path]
    for _ in range(rounds):
        if len(P) < 3:
            break
        i, j = sorted(rng.choice(len(P), 2, replace=False))
        if j - i < 2:
            continue
        if edge_ok(valid, P[i], P[j], step):
            P = P[: i + 1] + P[j:]
    i = 0
    while i < len(P) - 2:                                   # greedy final pass
        if edge_ok(valid, P[i], P[i + 2], step):
            del P[i + 1]
        else:
            i += 1
    return np.array(P)


def smooth(valid: Callable, path: np.ndarray, *, step: float = 0.01, iterations: int = 4,
           cut: float = 0.25) -> np.ndarray:
    """Chaikin corner cutting, ends kept; an iteration is kept only if every new edge is valid."""
    P = np.asarray(path, float)
    for _ in range(iterations):
        if len(P) < 3:
            break
        out = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            out += [a + (b - a) * cut, a + (b - a) * (1.0 - cut)]
        out = out[:1] + out[2:-1] + [P[-1]]                    # the first and last legs keep their ends
        Q = np.array(out)
        if not all(edge_ok(valid, x, y, step) for x, y in zip(Q[:-1], Q[1:])):
            break
        P = Q
    return P


def split_at_corners(Q: np.ndarray, min_cos: float = 0.9) -> list:
    """Pieces of a joint path between its sharp corners (direction change > acos(min_cos))."""
    Q = np.asarray(Q, float)
    D = np.diff(Q, axis=0)
    n = np.linalg.norm(D, axis=1)
    keep = n > 1e-12
    idx = np.flatnonzero(keep)
    cuts = [0]
    for i, j in zip(idx[:-1], idx[1:]):
        if float(D[i] @ D[j]) / (n[i] * n[j]) < min_cos:
            cuts.append(j)
    cuts.append(len(Q) - 1)
    return [Q[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

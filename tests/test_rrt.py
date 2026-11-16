"""CPU fallback planner for free moves (rsf.planning.rrt) on a 2-D toy."""
from __future__ import annotations

import numpy as np

from rsf.planning import rrt


def wall(q):
    """A wall x in [-0.2, 0.2] with a gap at y > 0.6, inside [-1, 1]^2."""
    return not (abs(q[0]) < 0.2 and q[1] < 0.6)


LO, HI = np.array([-1.0, -1.0]), np.array([1.0, 1.0])


def test_straight_when_free():
    p = rrt.rrt_connect(lambda q: True, [-0.5, 0.0], [0.5, 0.0], LO, HI)
    assert p.shape == (2, 2)


def test_around_the_wall_then_shorter_and_smooth():
    a, b = np.array([-0.8, -0.8]), np.array([0.8, -0.8])
    stats = {}
    p = rrt.rrt_connect(wall, a, b, LO, HI, step=0.01, stats=stats, seed=1)
    assert p is not None and np.allclose(p[0], a) and np.allclose(p[-1], b)
    assert all(rrt.edge_ok(wall, x, y, 0.01) for x, y in zip(p[:-1], p[1:]))
    s = rrt.shortcut(wall, p, step=0.01, seed=1)
    length = lambda P: float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())  # noqa: E731
    assert length(s) <= length(p) + 1e-9
    m = rrt.smooth(wall, s, step=0.01)
    assert np.allclose(m[0], a) and np.allclose(m[-1], b)
    assert all(rrt.edge_ok(wall, x, y, 0.01) for x, y in zip(m[:-1], m[1:]))
    assert all(wall(x) for x in m)


def test_no_path():
    blocked = lambda q: abs(q[0]) >= 0.2  # noqa: E731  -- a full wall
    assert rrt.rrt_connect(blocked, [-0.8, 0.0], [0.8, 0.0], LO, HI, max_iter=200) is None


def test_split_at_corners():
    Q = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], float)
    parts = rrt.split_at_corners(Q)
    assert [len(p) for p in parts] == [3, 3]
    assert np.allclose(parts[0][-1], parts[1][0])
    smooth = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 1, 50)])
    assert len(rrt.split_at_corners(smooth)) == 1

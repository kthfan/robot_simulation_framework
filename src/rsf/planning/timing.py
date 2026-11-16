"""Time profiles for moves (port of AI-Robotics-V2 program/motion/timing.py, D30). Pure numpy.

Straight-line moves follow a trapezoid on a unit path parameter, slowed until
both the linear and the angular caps hold. Joint-space paths (cuRobo's free
moves) are slowed by one uniform time scale. Checks of what a trajectory
actually does are finite differences, so they judge the samples that are sent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Caps:
    """Linear (m/s, m/s^2) and angular (rad/s, rad/s^2) limits of a moving frame."""

    v: float
    a: float
    w: float = math.inf
    alpha: float = math.inf
    a_down: float = math.inf


def caps_from(d: dict) -> Caps:
    """``{v, a, w, alpha, a_down}`` (missing: unlimited) as :class:`Caps`."""
    return Caps(float(d["v"]), float(d["a"]), float(d.get("w", math.inf)), float(d.get("alpha", math.inf)),
                float(d.get("a_down", math.inf)))


def trapezoid(length: float, v: float, a: float) -> float:
    """Duration of a rest-to-rest move of *length* at cruise *v*, acceleration *a*."""
    if length <= 0.0:
        return 0.0
    if length >= v * v / a:
        return length / v + v / a
    return 2.0 * math.sqrt(length / a)


def trapezoid_s(length: float, v: float, a: float, t) -> np.ndarray:
    """Distance travelled at times *t* along the :func:`trapezoid` move."""
    t = np.asarray(t, dtype=float)
    T = trapezoid(length, v, a)
    vp = min(v, math.sqrt(length * a))
    ta = vp / a
    s = np.where(t < ta, 0.5 * a * t * t,
                 np.where(t <= T - ta, 0.5 * a * ta * ta + vp * (t - ta),
                          length - 0.5 * a * (T - t) ** 2))
    return np.clip(s, 0.0, length)


def unit_timing(length: float, angle: float, caps: Caps, dt: float):
    """``(t, u)``: a uniform time grid and the path parameter 0..1 on it.

    *length* (m) and *angle* (rad) are what the moving frame covers between
    ``u = 0`` and ``u = 1``; the slower of the two sets of caps wins.
    """
    if length <= 0.0 and angle <= 0.0:
        raise ValueError("null move")
    v = min(caps.v / length if length > 0 else math.inf,
            caps.w / angle if angle > 0 else math.inf)
    a = min(caps.a / length if length > 0 else math.inf,
            caps.alpha / angle if angle > 0 else math.inf)
    T = trapezoid(1.0, v, a)
    n = max(1, math.ceil(T / dt))
    t = np.linspace(0.0, T, n + 1)
    return t, trapezoid_s(1.0, v, a, t)


def joint_retime(t, q, v_max: float, a_max: float) -> float:
    """Uniform time scale (>= 1) that brings every joint within the caps."""
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    qd = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    return max(1.0, float(np.abs(qd).max()) / v_max,
               math.sqrt(float(np.abs(qdd).max()) / a_max))


def resample(t, q, dt: float):
    """*q* linearly interpolated onto a uniform grid of about *dt*, ends kept."""
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    n = max(1, math.ceil((t[-1] - t[0]) / dt - 1e-9))
    t2 = np.linspace(t[0], t[-1], n + 1)
    q2 = np.column_stack([np.interp(t2, t, q[:, j]) for j in range(q.shape[1])])
    return t2 - t2[0], q2


def velocities(t, q) -> np.ndarray:
    """Joint velocities of the samples, zero at both ends (rest to rest)."""
    qd = np.gradient(np.asarray(q, dtype=float), np.asarray(t, dtype=float), axis=0)
    qd[0] = 0.0
    qd[-1] = 0.0
    return qd


def point_kinematics(t, P) -> dict:
    """Peak speed, acceleration and downward acceleration of a point path."""
    v = np.gradient(np.asarray(P, dtype=float), np.asarray(t, dtype=float), axis=0)
    a = np.gradient(v, t, axis=0)
    return {"v": float(np.linalg.norm(v, axis=1).max()),
            "a": float(np.linalg.norm(a, axis=1).max()),
            "a_down": float(max(0.0, -a[:, 2].min()))}


def _log_so3(R) -> np.ndarray:
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * th / (2.0 * math.sin(th))


def angular_kinematics(t, R) -> dict:
    """Peak angular speed and acceleration of a rotation path (world frame)."""
    t, R = np.asarray(t, dtype=float), np.asarray(R, dtype=float)
    dt = np.diff(t)
    w = np.array([_log_so3(R[i + 1] @ R[i].T) / dt[i] for i in range(len(dt))])
    tm = 0.5 * (t[1:] + t[:-1])
    al = np.gradient(w, tm, axis=0) if len(w) > 1 else np.zeros_like(w)
    return {"w": float(np.linalg.norm(w, axis=1).max()),
            "alpha": float(np.linalg.norm(al, axis=1).max())}

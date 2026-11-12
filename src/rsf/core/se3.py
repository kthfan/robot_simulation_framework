"""SE(3) / SO(3) helpers on plain 4x4 / 3x3 numpy arrays.

Conventions: metres, radians, quaternions ``wxyz``. URDF ``rpy`` is the fixed-axis
roll-pitch-yaw, ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""
from __future__ import annotations

import math

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def matrix_to_rpy(R) -> np.ndarray:
    """Inverse of :func:`rpy_to_matrix` (pitch in [-pi/2, pi/2])."""
    R = np.asarray(R, dtype=float)
    p = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(p)) < 1e-9:  # gimbal lock: put everything in yaw
        return np.array([0.0, p, math.atan2(-R[0, 1], R[1, 1])])
    return np.array([math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])])


def axis_angle(axis, angle: float) -> np.ndarray:
    """Rotation of *angle* about the unit vector *axis* (Rodrigues)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def make_T(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def from_xyz_rpy(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    return make_T(rpy_to_matrix(rpy), np.asarray(xyz, dtype=float))


def inv(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def log_so3(R) -> np.ndarray:
    """Rotation vector of *R*; accurate near 0 and near pi."""
    R = np.asarray(R, dtype=float)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(w))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    th = math.atan2(s, c)
    if th < 1e-9:
        return w
    if math.pi - th < 1e-4:
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / math.sqrt(max(A[k, k], 1e-300))
        if float(ax @ w) < 0.0:
            ax = -ax
        return ax * th
    return w * (th / s)


def exp_so3(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    return axis_angle(w / th, th)


def pose_error(Ta, Tb) -> tuple[float, float]:
    """``(metres, radians)`` between two poses."""
    Ta, Tb = np.asarray(Ta, dtype=float), np.asarray(Tb, dtype=float)
    dp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    dr = float(np.linalg.norm(log_so3(Ta[:3, :3].T @ Tb[:3, :3])))
    return dp, dr


def interpolate(T0, T1, u: float) -> np.ndarray:
    """Position linear, rotation along the geodesic; ``u`` in [0, 1]."""
    T0, T1 = np.asarray(T0, dtype=float), np.asarray(T1, dtype=float)
    R = T0[:3, :3] @ exp_so3(u * log_so3(T0[:3, :3].T @ T1[:3, :3]))
    return make_T(R, (1.0 - u) * T0[:3, 3] + u * T1[:3, 3])


def quat_wxyz(R) -> np.ndarray:
    """Unit quaternion ``wxyz`` of *R* (w >= 0)."""
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q_wxyz, dtype=float) / np.linalg.norm(q_wxyz))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

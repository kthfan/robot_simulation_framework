"""Pose construction and frame-convention conversion for the TM tray cell.

Ported verbatim from AI-Robotics-V2 ``cell_core/frames.py``. The assembled
``tm5s_egu50.urdf`` flange already uses the USD convention (its FK at q = 0 is
``USD_FLANGE_R0``), so :func:`flange_correction` of it is the identity.


The URDF and the USD describe the same physical flange with *different* basis
conventions, and nothing in either file says so. A sweep run without the
conversion below produced 222 plausible, internally consistent, entirely wrong
results (findings/04 §1.3). Keep the conversion explicit and keep it tested.
"""
from __future__ import annotations

import math

import numpy as np

#: World up.
UP = np.array([0.0, 0.0, 1.0])

#: The flange basis at the zero configuration as authored in ``tm5s.usd``,
#: columns being the world images of the flange axes. ``tm5s_v4.urdf`` instead
#: puts the flange at identity, hence :func:`flange_correction`.
USD_FLANGE_R0 = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]])


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, s],
                     [0.0, -s, c]])


def make_pose(R, p) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of quaternion *q* given as ``(x, y, z, w)``; normalises."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def usd_camera_attributes(fx, fy, cx, cy, width, height, focal_length) -> dict:
    """USD camera attributes that render the pinhole ``(fx, fy, cx, cy)``.

    USD has one focal length, so ``fx != fy`` is carried by the aperture
    aspect. *focal_length* only sets the unit scale of the apertures.
    ``(cx, cy)`` follow OpenCV (and the aprilgrid detector): pixel ``i`` is
    centred on ``i``. The USD film puts pixel ``i`` over ``[i, i+1]``, so the
    principal point on the film is ``cx + 0.5``.
    """
    ha = focal_length * width / fx
    va = focal_length * height / fy
    return {
        "focalLength": float(focal_length),
        "horizontalAperture": ha,
        "verticalAperture": va,
        "horizontalApertureOffset": (width / 2 - (cx + 0.5)) * ha / width,
        "verticalApertureOffset": ((cy + 0.5) - height / 2) * va / height,
    }


def intrinsics_from_usd(attrs, width, height) -> dict:
    """Inverse of :func:`usd_camera_attributes`."""
    f, ha, va = attrs["focalLength"], attrs["horizontalAperture"], attrs["verticalAperture"]
    return {
        "fx": f * width / ha,
        "fy": f * height / va,
        "cx": width / 2 - attrs["horizontalApertureOffset"] * width / ha - 0.5,
        "cy": height / 2 + attrs["verticalApertureOffset"] * height / va - 0.5,
    }


def flange_rotation(approach, tilt: float = 0.0) -> np.ndarray:
    """Rotation whose columns are the world images of the flange axes.

    ``flange Z`` -> *approach*, a horizontal vector pointing into the shelf
    ``flange Y`` -> world up, so the plate hangs below the tray and stays level
    ``flange X`` -> ``up x approach``

    A positive *tilt* rotates about the flange X axis so the plate tip rises,
    which is how the tray is retained gravitationally during carry (D4/D14).
    """
    a = np.asarray(approach, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("approach vector must be non-zero")
    a = a / n
    lateral = np.cross(UP, a)
    lat_n = np.linalg.norm(lateral)
    if lat_n < 1e-9:
        raise ValueError("approach vector must be horizontal, not vertical")
    R = np.column_stack([lateral / lat_n, UP, a])
    if tilt:
        R = R @ rot_x(tilt)
    return R


def lateral_axis(approach) -> np.ndarray:
    """Horizontal axis perpendicular to *approach*, i.e. along the rack width."""
    return flange_rotation(approach)[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# URDF <-> USD flange conventions
# ─────────────────────────────────────────────────────────────────────────────
def flange_correction(urdf_flange_R0) -> np.ndarray:
    """Constant ``A`` relating the two flange conventions.

    Defined by ``R_usd(q) = R_urdf(q) @ A``, so that

        ``A = R_urdf(0).T @ USD_FLANGE_R0``

    Pass the rotation part of the URDF chain's forward kinematics at ``q = 0``.
    For ``tm5s_v4.urdf`` that is the identity, making ``A == USD_FLANGE_R0``, but
    it is computed rather than assumed so that swapping the URDF cannot silently
    invalidate every pose.
    """
    R0 = np.asarray(urdf_flange_R0, dtype=float)
    if R0.shape != (3, 3):
        raise ValueError("expected a 3x3 rotation matrix")
    if not np.allclose(R0 @ R0.T, np.eye(3), atol=1e-9):
        raise ValueError("expected an orthonormal rotation matrix")
    return R0.T @ USD_FLANGE_R0


def usd_to_urdf(T, A) -> np.ndarray:
    """Re-express a USD-convention flange pose in the URDF convention."""
    out = np.array(T, dtype=float, copy=True)
    out[:3, :3] = out[:3, :3] @ np.asarray(A, dtype=float).T
    return out


def urdf_to_usd(T, A) -> np.ndarray:
    """Re-express a URDF-convention flange pose in the USD convention."""
    out = np.array(T, dtype=float, copy=True)
    out[:3, :3] = out[:3, :3] @ np.asarray(A, dtype=float)
    return out


def pose_error(a, b) -> tuple[float, float]:
    """Position and rotation error between two 4x4 poses.

    Returns ``(metres, radians)``. The rotation term is the geodesic angle, so
    it is comparable across axes rather than being a matrix norm.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dp = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return dp, float(math.acos(max(-1.0, min(1.0, c))))

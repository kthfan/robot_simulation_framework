"""Locating: the detector's board pose in the world, and whether it is good enough.

The detector (``aprilgrid_detector.detect_boards``) returns the board in the
camera frame as OpenCV does: ``rvec`` and ``translation`` map board points into
the camera. Chained with the camera's world pose, that is the board in the
world. The error against a reference pose is split along the slot's own axes,
because the tolerance budget is: vertical and pitch against the corridor
height, lateral and yaw against the pocket width (D17), depth against the
insertion margins (D27).
"""
from __future__ import annotations

import math

import numpy as np

from . import frames as F
from . import geometry as G
from . import sequence as S

#: The axes :func:`failures` judges, per read; the rest are reported only.
#: Both reads are judged on all but roll (D27, superseding D25's far-read
#: exemption): the D26 renders resolve far pitch and depth.
JUDGED = {"far": ("vertical", "lateral", "depth", "pitch", "yaw"),
          "near": ("vertical", "lateral", "depth", "pitch", "yaw")}

#: A board fitted worse than this is reported as ``high_rmse``, not trusted.
#: User decision (D25): the strict renders fit at 0.80-1.34 px from texture
#: filtering, with sub-0.05 mm positions. Not a measured spec.
MAX_RMSE_PX = 2.0


def choose_board(boards, grid_id):
    """The slot's own board among what the detector saw, and why not if none.

    Returns ``(board, status)``. At the far read the neighbouring jigs are in
    frame, so the board is picked by id. ``status`` is ``ok``, ``high_rmse``
    (board returned, but its fit is poor), or with no board: ``missing``
    (nothing seen), ``wrong_id`` (only other boards), ``duplicate`` (the id
    twice) or ``no_pose``.
    """
    mine = [b for b in boards if b.board_id == grid_id]
    if len(mine) > 1:
        return None, "duplicate"
    if not mine:
        return None, "wrong_id" if boards else "missing"
    b = mine[0]
    if b.pose is None:
        return None, "no_pose"
    if b.pose.reprojection_rmse > MAX_RMSE_PX:
        return b, "high_rmse"
    return b, "ok"


def rodrigues(rvec) -> np.ndarray:
    """Rotation matrix of rotation vector *rvec* (as ``cv2.Rodrigues``)."""
    r = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    if theta < 1e-12:
        return np.eye(3) + K
    K /= theta
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * K @ K


def _rotvec(R) -> np.ndarray:
    """Inverse of :func:`rodrigues`, for rotations well short of 180 degrees."""
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = float(np.linalg.norm(v))
    c = (np.trace(R) - 1.0) / 2.0
    theta = math.atan2(s, c)
    return v if s < 1e-12 else v * (theta / s)


def board_in_camera(translation, rvec) -> np.ndarray:
    """The detector's pose as a 4x4: board frame -> camera optical frame."""
    return F.make_pose(rodrigues(rvec), translation)


def board_in_world(camera_T, board_in_camera_T) -> np.ndarray:
    return np.asarray(camera_T, dtype=float) @ np.asarray(board_in_camera_T, dtype=float)


def pose_errors(measured_T, reference_T, approach) -> dict:
    """*measured* minus *reference*, along the slot axes of *approach*.

    Positions in metres at the board origin: ``vertical`` (world up),
    ``lateral`` (:func:`frames.lateral_axis`), ``depth`` (along the approach).
    Angles in degrees, of the world-frame rotation taking reference to
    measured: ``pitch`` about the lateral axis, ``yaw`` about up, ``roll``
    about the approach.
    """
    m = np.asarray(measured_T, dtype=float)
    r = np.asarray(reference_T, dtype=float)
    a = np.asarray(approach, dtype=float)
    a = a / np.linalg.norm(a)
    lat = F.lateral_axis(a)
    d = m[:3, 3] - r[:3, 3]
    w = np.degrees(_rotvec(m[:3, :3] @ r[:3, :3].T))
    return {
        "vertical": float(d @ F.UP), "lateral": float(d @ lat), "depth": float(d @ a),
        "pitch": float(w @ lat), "yaw": float(w @ F.UP), "roll": float(w @ a),
    }


def limits() -> dict:
    """Half of each budget (M4 acceptance).

    Vertical, lateral, pitch, yaw: the flat corridor (D17 designs to no
    recess). Depth (D27): a depth error moves the plate tip along the approach,
    into the pre-insert gap and the back margin, and (D28) the gripper body
    into the front margin, so the tightest of the three.
    """
    b = G.tolerance_budget(use_recess=False)
    return {"vertical": b["vertical_half"] / 2, "lateral": b["lateral_half"] / 2,
            "depth": min(S.BACK_MARGIN, S.PRE_INSERT_GAP, S.FRONT_MARGIN) / 2,
            "pitch": b["pitch_deg"] / 2, "yaw": b["yaw_deg"] / 2}


def failures(errors, lim, read) -> list:
    """The axes judged for *read* whose error exceeds its limit, in :data:`JUDGED` order."""
    return [k for k in JUDGED[read] if abs(errors[k]) > lim[k]]

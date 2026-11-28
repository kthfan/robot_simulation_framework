"""What the locating camera sees (P6): every slot's AprilGrid label and the EIH camera.

* :func:`labels`: per slot, the printed AprilGrid (4 x 1 tags of 36h11, 20 mm,
  7.5 mm apart, 5 mm white margin: 112.5 x 30 mm, the jig's label board) on
  the label's robot-facing surface, its grid id (``geometry.grid_ids``: marker
  ids ``4 * grid_id + 0..3``), the sheet image, and ``board_T``: the
  detector's board frame in the world (origin at tag 0's top-left corner, x
  along the printed columns, y down the rows, z into the label; as
  AI-Robotics-V2 M4 defined it).
* :func:`locating_camera`: the EIH 5M camera, hand-eye calibrated on the
  flange (``hardware.CAM_*``), with the intrinsics of the focus config nearest
  the read distance (``hardware.FOCAL_CONFIGS``).
* :func:`reads` (``process.reads: tm_tray.reads``): the locate tour's reads for
  :mod:`rsf.sim.reads`, with the detector (:func:`detect_boards`, when
  ``aprilgrid_detector`` is installed).
"""
from __future__ import annotations

import math
import re

import numpy as np

from rsf.core import se3
from rsf.model.registry import register

from . import hardware as A
from .geometry import grid_ids, slot_of

MARGIN = 0.005                       #: white border around the grid (the label is 112.5 x 30 mm)
SHEET_PX_PER_M = 8000.0              #: rasterised sheet: 160 px per 20 mm tag


def sheet_image(grid_id: int, px_per_m: float = SHEET_PX_PER_M) -> np.ndarray:
    """The printed label: tags ``4 * grid_id + i`` left to right, white margin around."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    w = A.GRID_W + 2 * MARGIN
    h = A.GRID_H + 2 * MARGIN
    img = np.full((int(round(h * px_per_m)), int(round(w * px_per_m))), 255, np.uint8)
    n = int(round(A.TAG_SIZE * px_per_m))
    for i in range(A.TAGS_X * A.TAGS_Y):
        row, col = divmod(i, A.TAGS_X)
        x0 = int(round((MARGIN + col * (A.TAG_SIZE + A.TAG_SEPARATION)) * px_per_m))
        y0 = int(round((MARGIN + row * (A.TAG_SIZE + A.TAG_SEPARATION)) * px_per_m))
        img[y0:y0 + n, x0:x0 + n] = cv2.aruco.generateImageMarker(d, A.TAGS_X * A.TAGS_Y * grid_id + i, n)
    return img


@register("vision", "tm_tray.labels")
def labels(world) -> list:
    """Every slot's label: ``{key, grid_id, T_sheet, size, image, board_T}`` (world frame).

    ``T_sheet``: the sheet centre, x right and y up as the robot reads it, z
    toward the robot (``rsf.sim.vision.add_marker``)."""
    feats = world.features({"kind": "slot"})
    ids = grid_ids([slot_of(f) for f in feats])
    (cx, cy, cz), (sx, sy, sz) = A.JIG_BOXES["label"]
    w, h = A.GRID_W + 2 * MARGIN, A.GRID_H + 2 * MARGIN
    # jig frame: x lateral, y toward the robot, z up. Seen from the robot: right = -x, up = +z
    R_sheet = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    T_jig_sheet = se3.make_T(R_sheet, (cx, cy + sy / 2 + 1e-4, cz))
    # board frame on the sheet: origin at tag 0's top-left, x right, y down, z into the label
    T_sheet_board = se3.make_T(np.diag([1.0, -1.0, -1.0]), (-w / 2 + MARGIN, h / 2 - MARGIN, 0.0))
    out = []
    for f in feats:
        T = np.asarray(f.T) @ T_jig_sheet
        out.append({"key": f.key, "grid_id": ids[f.key], "T_sheet": T, "size": (w, h),
                    "image": None, "board_T": T @ T_sheet_board})
    return out


def focal_config(distance: float) -> dict:
    k = A.focal_config_for(distance)
    return {"config": k, **{n: A.FOCAL_CONFIGS[k][n] for n in ("fx", "fy", "cx", "cy")},
            "width": A.EIH_WIDTH, "height": A.EIH_HEIGHT}


@register("vision", "tm_tray.camera")
def locating_camera(world, distance: float) -> dict:
    """``{link, T_link_cv, fx, fy, cx, cy, width, height, config}`` for a read at *distance* (m)."""
    x, y, z, w = A.CAM_QUAT
    R = se3.quat_to_matrix((w, x, y, z))
    return {"link": "flange", "T_link_cv": se3.make_T(R, (A.CAM_FX, A.CAM_FY, A.CAM_FZ)), **focal_config(distance)}



#: the reads of a locate sequence (``tm_tray.locate``) and the standoff each is taken at
READS = {"sight_far": "standoff_far", "sight_near": "standoff_near"}
_READ_SEG = re.compile(r"^locate:(.+)/(sight_far|sight_near):act$")


def _rodrigues(rvec) -> np.ndarray:
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rvec, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def detect_boards(grey, fx, fy, cx, cy) -> dict:
    """``{grid_id: T_camera_board}`` of every AprilGrid label the user's detector
    (kthfan/AprilgridDetector) finds in *grey*."""
    import aprilgrid_detector as agd
    det = agd.Detector()
    layout = agd.BoardLayout(tags_x=A.TAGS_X, tags_y=A.TAGS_Y, tag_size=A.TAG_SIZE,
                             tag_separation=A.TAG_SEPARATION)
    out = {}
    for b in det.detect_boards(np.ascontiguousarray(grey), layout, agd.Camera(fx=fx, fy=fy, cx=cx, cy=cy)):
        T = np.eye(4)
        T[:3, :3] = _rodrigues(np.asarray(b.pose.rvec, float))
        T[:3, 3] = np.asarray(b.pose.translation, float)
        out[int(b.board_id)] = T
    return out


@register("vision", "tm_tray.reads")
def reads(world):
    """The locate tour's reads (``rsf.sim.reads``): the EIH camera at each read's focus
    config, every slot's label, the detector, and the near reads to fit a rack with."""
    import importlib.util

    from rsf.sim.reads import Marker, ReadCamera, ReadSetup
    cams, which = {}, {}
    for read, key in READS.items():
        c = locating_camera(world, float(world.process[key]))
        name = f"eih_c{c['config']}"
        cams[name] = ReadCamera(name, c["link"], c["T_link_cv"], c["fx"], c["fy"], c["cx"], c["cy"],
                                c["width"], c["height"])
        which[read] = name
    ent = {f.key: f.entity for f in world.features({"kind": "slot"})}
    markers = [Marker(l["key"], ent[l["key"]], l["T_sheet"], l["size"], sheet_image(l["grid_id"]), l["board_T"],
                      l["grid_id"]) for l in labels(world)]

    def match(name: str):
        m = _READ_SEG.match(name)
        return (m.group(1), m.group(2), which[m.group(2)]) if m else None

    corners = np.array([[0.0, 0.0, 0.0, 1.0], [A.GRID_W, 0.0, 0.0, 1.0], [0.0, A.GRID_H, 0.0, 1.0],
                        [A.GRID_W, A.GRID_H, 0.0, 1.0]]).T
    has_detector = importlib.util.find_spec("aprilgrid_detector") is not None
    return ReadSetup(cams, markers, match, detect_boards if has_detector else None, ("sight_near",), corners)

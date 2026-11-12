"""The cell as collision boxes, for cuRobo (M3).

Built from the same :class:`~cell_core.config.CellConfig` the reach sweep is
judged on (D18), so the collision verdicts describe the same cell. Every solid
is a box:

* shelves, the table top and the pedestal base are boxes in the builder too;
* the jig is all boxes (``assets.JIG_BOXES``, copied from its USD asset);
* the tray is its bounding box, ending at the rail plane rather than at the
  2.5 mm recess (D17 designs to a flat underside);
* cylinders (rack legs, pedestal legs and plate) are their enclosing boxes,
  which overstates them by at most a few millimetres, on the safe side.

Positions are world frame (z up, floor at 0, robot axis at x = y = 0) until
:func:`to_curobo_scene` re-expresses them in the robot base frame, with cuRobo's
``[x, y, z, qw, qx, qy, qz]`` pose order.

Rack widths are the widths the builder produces, which can differ from the
configured ones (``env_adapter.unsupported`` reports by how much), because that
is the stage the measured-stage test compares against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from . import assets as A
from . import frames as F
from . import sequence as SEQ
from .config import CellConfig, min_rack_width
from .env_adapter import generator_kwargs
from .geometry import Slot, enumerate_slots

#: Phases in which the target tray rides on the plate instead of its jig.
#: Release visits the same poses (``sequence.release_sequence``) with the same
#: split: the tray is on the plate until ``insert`` puts it down.
CARRIED_PHASES = ("lift", "withdraw", "carry")

#: The floor, as a slab whose top face is z = 0.
FLOOR_SIZE = (6.0, 6.0, 0.1)

#: Largest turn of a found jig from its slot. A slot carries a position only,
#: so a jig turned further cannot be planned in (the placement limit, 1 deg).
LOCATED_TURN_TOL = math.radians(1.0)


@dataclass(frozen=True, eq=False)
class Box:
    """An oriented box: *T* places its centre, *size* is its full extent per local axis."""

    name: str
    T: np.ndarray
    size: np.ndarray

    def corners(self) -> np.ndarray:
        """The eight corners, world frame, shape ``(8, 3)``."""
        half = np.asarray(self.size, dtype=float) / 2.0
        local = np.array(list(product((-1.0, 1.0), repeat=3))) * half
        return local @ self.T[:3, :3].T + self.T[:3, 3]


def _box(name: str, R, centre, size) -> Box:
    return Box(name, F.make_pose(R, centre), np.asarray(size, dtype=float))


def separation(a: Box, b: Box) -> float:
    """Largest gap between *a* and *b* along any separating-axis candidate.

    Positive: the boxes are disjoint and at least this far apart (it is a lower
    bound of their distance, exact when a face separates them). Negative: they
    overlap, by about this much.
    """
    Ra, Rb = a.T[:3, :3], b.T[:3, :3]
    ha, hb = np.asarray(a.size) / 2.0, np.asarray(b.size) / 2.0
    d = b.T[:3, 3] - a.T[:3, 3]
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(Ra[:, i], Rb[:, j])
            n = np.linalg.norm(c)
            if n > 1e-9:
                axes.append(c / n)
    best = -math.inf
    for L in axes:
        ra = float(np.sum(ha * np.abs(Ra.T @ L)))
        rb = float(np.sum(hb * np.abs(Rb.T @ L)))
        best = max(best, abs(float(d @ L)) - ra - rb)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Jigs and trays
# ─────────────────────────────────────────────────────────────────────────────
def jig_pose(slot: Slot) -> np.ndarray:
    """World pose of the jig origin: jig +y faces the robot, +z is up."""
    a = slot.approach_vec / np.linalg.norm(slot.approach_vec)
    lat = F.lateral_axis(a)
    face = slot.label_plane_dist + A.LABEL_PROUD_OF_FACE
    centre = a * (face + A.JIG_Y_OFFSET) + lat * slot.lateral + F.UP * slot.jig_origin_z
    return F.make_pose(np.column_stack([lat, -a, F.UP]), centre)


def located(slot: Slot, jig_T) -> Slot:
    """*slot* moved so that its jig stands at *jig_T*: the inverse of :func:`jig_pose`.

    M6 (D30): plans follow the jig that was found, not the nominal one. Raises
    if the found jig is turned more than :data:`LOCATED_TURN_TOL`.
    """
    T = np.asarray(jig_T, dtype=float)
    _, turn = F.pose_error(T, jig_pose(slot))
    if turn > LOCATED_TURN_TOL:
        raise ValueError(f"{slot.key}: jig found turned {math.degrees(turn):.2f} deg")
    a = slot.approach_vec / np.linalg.norm(slot.approach_vec)
    p = T[:3, 3]
    return replace(slot,
                   lateral=float(F.lateral_axis(a) @ p),
                   label_plane_dist=float(a @ p) - A.LABEL_PROUD_OF_FACE - A.JIG_Y_OFFSET,
                   height=float(p[2]) - slot.plane_thickness / 2.0)


def _in_jig(slot: Slot, name: str, centre, size) -> Box:
    T = jig_pose(slot) @ F.make_pose(np.eye(3), centre)
    return Box(name, T, np.asarray(size, dtype=float))


def jig_boxes(slot: Slot) -> list:
    return [_in_jig(slot, f"jig/{slot.key}/{part}", c, s)
            for part, (c, s) in A.JIG_BOXES.items()]


def tray_box(slot: Slot) -> Box:
    """The tray resting on its jig's columns."""
    return _in_jig(slot, f"tray/{slot.key}",
                   (0.0, 0.0, A.TRAY_BOT + A.TRAY_Z / 2.0),
                   (A.TRAY_X, A.TRAY_Y, A.TRAY_Z))


def carried_tray_box(flange_T, name: str = "tray/carried") -> Box:
    """The tray riding on the plate, recess on the plate's upper face.

    Its depth along the plate is where :func:`sequence.insert_pose` picked it
    up: centred on the jig, the plate tip at :func:`sequence.insert_tip_depth`.
    """
    centre = (0.0,
              A.PLATE_FY_TOP - A.TRAY_RECESS + A.TRAY_Z / 2.0,
              A.PLATE_FZ_TIP + A.JIG_Y_OFFSET - SEQ.insert_tip_depth())
    T = np.asarray(flange_T, dtype=float) @ F.make_pose(np.eye(3), centre)
    return Box(name, T, np.array([A.TRAY_X, A.TRAY_Z, A.TRAY_Y]))


# ─────────────────────────────────────────────────────────────────────────────
# Racks, table, pedestal, floor
# ─────────────────────────────────────────────────────────────────────────────
def _facing(approach) -> np.ndarray:
    """Box axes for something facing the robot: x along *approach*, y lateral, z up."""
    a = np.asarray(approach, dtype=float)
    a = a / np.linalg.norm(a)
    return np.column_stack([a, F.lateral_axis(a), F.UP])


def rack_boxes(cfg: CellConfig) -> list:
    kw = generator_kwargs(cfg)
    boxes = []
    for rack in cfg.racks:
        R = _facing(rack.approach)
        a, lat = R[:, 0], R[:, 1]
        width = (min_rack_width(rack.n_jigs, kw["jig_gap"], kw["cylinder_radius"])
                 + kw["size_x_margin"])
        r, t, depth = rack.cylinder_radius, rack.plane_thickness, rack.depth
        mid = rack.face_distance + depth / 2.0
        for i, h in enumerate(rack.layer_heights):
            boxes.append(_box(f"rack/{rack.name}/shelf{i}", R,
                              a * mid + F.UP * h, (depth, width, t)))
        height = max(rack.layer_heights) + t
        legs = product((rack.face_distance + r, rack.face_distance + depth - r),
                       (-(width / 2.0 - r), width / 2.0 - r))
        for i, (d, u) in enumerate(legs):
            boxes.append(_box(f"rack/{rack.name}/leg{i}", R,
                              a * d + lat * u + F.UP * (height / 2.0),
                              (2 * r, 2 * r, height)))
    return boxes


def table_boxes(cfg: CellConfig) -> list:
    t = cfg.table
    R = _facing(t.approach)
    centre = R[:, 0] * (t.face_distance + t.depth / 2.0) + F.UP * (t.height / 2.0)
    return [_box("table/top", R, centre, (t.depth, A.TABLE_WIDTH, t.height))]


def mount_boxes(cfg: CellConfig) -> list:
    m = cfg.robot
    I = np.eye(3)
    r = A.MOUNT_LEG_RADIUS
    span = A.MOUNT_LEG_GAP / 2.0 + r
    boxes = [_box("mount/base", I, (0.0, 0.0, m.base_size / 2.0),
                  (A.MOUNT_BASE_XY, A.MOUNT_BASE_XY, m.base_size))]
    for i, (sx, sy) in enumerate(product((-1.0, 1.0), repeat=2)):
        boxes.append(_box(f"mount/leg{i}", I,
                          (sx * span, sy * span, m.base_size + m.leg_height / 2.0),
                          (2 * r, 2 * r, m.leg_height)))
    plate_r = math.sqrt(2.0) * span + r
    boxes.append(_box("mount/plate", I,
                      (0.0, 0.0, m.origin_z - m.plate_thickness / 2.0),
                      (2 * plate_r, 2 * plate_r, m.plate_thickness)))
    return boxes


def floor_box() -> Box:
    return _box("floor", np.eye(3), (0.0, 0.0, -FLOOR_SIZE[2] / 2.0), FLOOR_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# Whole cell
# ─────────────────────────────────────────────────────────────────────────────
def cell_boxes(cfg: CellConfig, empty=(), slots=None) -> list:
    """Everything in the cell, with a tray in every slot not named in *empty*.

    Every slot full is the worst case for a neighbour, whatever the schedule.
    *slots* (default: the config's) places the jigs, e.g. :func:`located` ones.
    """
    empty = set(empty)
    boxes = rack_boxes(cfg) + table_boxes(cfg) + mount_boxes(cfg) + [floor_box()]
    for slot in enumerate_slots(cfg) if slots is None else slots:
        boxes += jig_boxes(slot)
        if slot.key not in empty:
            boxes.append(tray_box(slot))
    return boxes


def phase_scene(cfg: CellConfig, slot: Slot, phase: str):
    """The world during *phase* at *slot*, and the tray on the plate if any.

    Returns ``(boxes, carried)``. From ``lift`` on, the target tray leaves the
    world and rides on the plate as *carried*; before that *carried* is None.
    """
    if phase not in SEQ.GRASP_PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    if phase in CARRIED_PHASES:
        T = SEQ.grasp_sequence(cfg, slot)[phase]
        return cell_boxes(cfg, empty=(slot.key,)), carried_tray_box(T)
    return cell_boxes(cfg), None


# ─────────────────────────────────────────────────────────────────────────────
# cuRobo form
# ─────────────────────────────────────────────────────────────────────────────
def _quat_wxyz(R) -> list:
    """Unit quaternion of rotation *R*, ``[w, x, y, z]`` with ``w >= 0``."""
    R = np.asarray(R, dtype=float)
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0,
                      R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0,
                      R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0,
                      R[1, 0] - R[0, 1])
    return [w, x, y, z]


def to_curobo_scene(boxes, base) -> dict:
    """``{"cuboid": {name: {"dims", "pose"}}}`` in the robot base frame.

    *base* is the 4x4 world pose of the arm's base (``RobotMount.base_pose``).
    Poses are ``[x, y, z, qw, qx, qy, qz]``.
    """
    inv = np.linalg.inv(np.asarray(base, dtype=float))
    cuboids = {}
    for b in boxes:
        if b.name in cuboids:
            raise ValueError(f"duplicate obstacle name {b.name!r}")
        T = inv @ b.T
        cuboids[b.name] = {
            "dims": [float(v) for v in b.size],
            "pose": [float(v) for v in T[:3, 3]] + _quat_wxyz(T[:3, :3]),
        }
    return {"cuboid": cuboids}

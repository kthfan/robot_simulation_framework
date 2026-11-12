"""Grasp phase sequencing: the flange poses that service one slot.

Ported from AI-Robotics-V2 ``cell_core/sequence.py``. The only change: the
sequence functions take the world's ``process`` mapping (``standoff_far``,
``standoff_near``, ``carry_tilt_deg``) instead of a ``CellConfig``.


The first six phases are exactly what ``tools/reach_sweep.py`` swept in
findings/04. If they change, that sweep no longer describes this system and must
be re-run.

**Release is not a separate set of poses.** The four mechanical poses form a
rectangle in (depth, height):

==============  ==========  ==========
                shallow     deep
==============  ==========  ==========
**high**        withdraw    lift
**low**         pre_insert  insert
==============  ==========  ==========

Grasping runs low-shallow -> low-deep -> high-deep -> high-shallow. Releasing
runs the same rectangle backwards, putting the tray down where grasping picked
it up. So :func:`release_sequence` reorders :func:`grasp_sequence`'s poses
rather than deriving its own; two independently derived sets could agree today
and drift apart later with nothing to catch it.

(An earlier version of this docstring said the release sequence was deliberately
absent because its poses had not been verified. That was wrong: the sweep had
already covered them, because they are these poses.)
"""
from __future__ import annotations

import math

import numpy as np

from . import hardware as A
from . import frames as F
from . import geometry as G
from .geometry import Slot

#: Ordered phases of picking a tray out of a jig.
GRASP_PHASES = (
    "sight_far",    # read the AprilGrid from a safe distance
    "sight_near",   # read it again, closer, for a tighter pose estimate
    "pre_insert",   # plate aligned to the jig corridor, just outside the shelf
    "insert",       # plate fully under the tray
    "lift",         # tray off its columns
    "withdraw",     # back out of the shelf, tray carried
    "carry",        # tip the plate back to retain the tray in transit
)

#: Ordered phases of placing a tray into a jig: the rectangle, reversed.
#:
#: No sighting phases. The slot was read on the way in, and there is no label to
#: re-read while holding a tray in front of it.
RELEASE_PHASES = (
    "carry",        # arrive still tilted, tray retained
    "withdraw",     # level the plate, high and clear of the shelf
    "lift",         # move in, still above the columns
    "insert",       # lower, setting the tray down on the columns
    "pre_insert",   # retract, plate now empty
)

#: Gap kept between the tray rails and the column tops at ``lift``, with the
#: plate located as low as the M4 vertical limit allows (D28).
LIFT_CLEARANCE = 0.005
#: How far the tray is raised off the jig columns (D28, was 0.040): the plate
#: first closes the gap to the tray's recess, then may sit a full M4 vertical
#: limit (``locating.limits``) low, then keeps ``LIFT_CLEARANCE``. Every
#: millimetre more is taken from the clearance under the shelf above.
LIFT_HEIGHT = (G.tolerance_budget(use_recess=True)["vertical_half"]
               + G.tolerance_budget(use_recess=False)["vertical_half"] / 2
               + LIFT_CLEARANCE)
#: Clearance kept between the plate tip and the back edge of the tray.
BACK_MARGIN = 0.010
#: Clearance kept between the gripper body and the jig in front of it (D28).
#: Equal to BACK_MARGIN: a depth error eats either one alike.
FRONT_MARGIN = 0.010
#: Gap between the plate tip and the label plane when staged.
PRE_INSERT_GAP = 0.010


def _pose(slot: Slot, dist: float, z: float, tilt: float = 0.0) -> np.ndarray:
    """Flange pose at *dist* along the approach axis and world height *z*."""
    a = slot.approach_vec
    a = a / np.linalg.norm(a)
    lateral = F.lateral_axis(a)
    return F.make_pose(F.flange_rotation(a, tilt),
                       a * dist + lateral * slot.lateral + F.UP * z)


def sighting_pose(slot: Slot, standoff: float, rise: float = 0.0) -> np.ndarray:
    """Put the EIH camera *standoff* in front of the label, level with it.

    *rise* lifts the camera above the label centre. It buys lower-layer
    clearance one-for-one but costs image centring (D24: the close read now
    clears the lower tray at the plate-free standoff without it).
    """
    camera_dist = slot.label_plane_dist - standoff
    return _pose(slot, camera_dist - A.CAM_FZ,
                 slot.label_centre_z + rise - A.CAM_FY)


def eih_camera_pose(flange_T) -> np.ndarray:
    """World pose of the EIH optical frame (+Z forward, +Y image-down).

    The flange pose composed with the hand-eye calibration, rotation included,
    so it is the camera the detector's board pose is relative to.
    """
    hand_eye = F.make_pose(F.quat_to_matrix(A.CAM_QUAT),
                           (A.CAM_FX, A.CAM_FY, A.CAM_FZ))
    return np.asarray(flange_T, dtype=float) @ hand_eye


def insertion_flange_z(slot: Slot) -> float:
    """Flange height that centres the plate in the jig pocket corridor."""
    return slot.jig_origin_z + A.CORRIDOR_MID - A.PLATE_MID_FY


def _jig_front_depth() -> float:
    """Depth behind the rack face of the first thing the gripper body meets.

    The body is wider than the pocket, so the front columns stop it, or the
    tray front if that ever stood further out.
    """
    cols = [A.JIG_Y_OFFSET - (c[1] + s[1] / 2.0)
            for part, (c, s) in A.JIG_BOXES.items()
            if part.startswith("col") and c[1] > 0]
    return min(cols + [A.TRAY_FRONT_DEPTH])


def insert_tip_depth(back_margin: float = BACK_MARGIN,
                     front_margin: float = FRONT_MARGIN) -> float:
    """Plate tip depth behind the rack face when fully inserted.

    *back_margin* short of the tray's back edge, unless the gripper body would
    then come within *front_margin* of the jig: then it stops there (user
    2026-10-06, D28: "if touched, stop it earlier").
    """
    back = A.TRAY_BACK_DEPTH - back_margin
    body = _jig_front_depth() - front_margin - A.BODY_FZ_FRONT + A.PLATE_FZ_TIP
    return min(back, body)


def insert_pose(slot: Slot, back_margin: float = BACK_MARGIN,
                front_margin: float = FRONT_MARGIN) -> np.ndarray:
    """Fully inserted, the tip at :func:`insert_tip_depth`."""
    tip_dist = (slot.label_plane_dist + A.LABEL_PROUD_OF_FACE
                + insert_tip_depth(back_margin, front_margin))
    return _pose(slot, tip_dist - A.PLATE_FZ_TIP, insertion_flange_z(slot))


def pre_insert_pose(slot: Slot, gap: float = PRE_INSERT_GAP) -> np.ndarray:
    """Staged in front of the shelf with the plate already on the corridor line."""
    tip_dist = slot.label_plane_dist - gap
    return _pose(slot, tip_dist - A.PLATE_FZ_TIP, insertion_flange_z(slot))


def lift_pose(slot: Slot, lift: float = LIFT_HEIGHT) -> np.ndarray:
    T = insert_pose(slot)
    T[2, 3] += lift
    return T


def withdraw_pose(slot: Slot, lift: float = LIFT_HEIGHT,
                  gap: float = PRE_INSERT_GAP) -> np.ndarray:
    T = pre_insert_pose(slot, gap)
    T[2, 3] += lift
    return T


def carry_pose(slot: Slot, tilt_deg: float = 0.0,
               lift: float = LIFT_HEIGHT,
               gap: float = PRE_INSERT_GAP) -> np.ndarray:
    """The withdraw pose with the plate tipped back by *tilt_deg* degrees.

    The flange pivots in place: only the rotation changes, so a tilt sweep
    measures the tilt and not an accidental change in how far the arm is
    reaching.

    Tilting *here* rather than anywhere earlier is deliberate. At withdraw the
    plate is already clear of the shelf face, so raising the tip has nothing
    above it to foul. Tilting at ``lift``, while the plate is still inside the
    shelf, would put the tip under the tray on the layer above.
    """
    T = withdraw_pose(slot, lift, gap)
    a = slot.approach_vec
    T[:3, :3] = F.flange_rotation(a / np.linalg.norm(a), math.radians(tilt_deg))
    return T


def grasp_sequence(process: dict, slot: Slot, near_rise: float = 0.0) -> dict:
    """The seven flange poses that pick the tray out of *slot*, in order.

    Standoffs and the carry tilt come from *process*, so a cell rebuilt with
    different optics, clearances or retention produces different poses without
    touching this module.
    """
    return {
        "sight_far": sighting_pose(slot, process["standoff_far"]),
        "sight_near": sighting_pose(slot, process["standoff_near"], near_rise),
        "pre_insert": pre_insert_pose(slot),
        "insert": insert_pose(slot),
        "lift": lift_pose(slot),
        "withdraw": withdraw_pose(slot),
        "carry": carry_pose(slot, process.get("carry_tilt_deg", 0.0)),
    }


def release_sequence(process: dict, slot: Slot, near_rise: float = 0.0) -> dict:
    """The flange poses that place a tray into *slot*, in order.

    Literally :func:`grasp_sequence`'s poses, reordered. See the module
    docstring for why this is a reordering and not a derivation.
    """
    poses = grasp_sequence(process, slot, near_rise)
    return {phase: poses[phase] for phase in RELEASE_PHASES}

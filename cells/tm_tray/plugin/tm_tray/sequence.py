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
from dataclasses import dataclass
from typing import Optional

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
    2026-10-06, D28: "if touched, stop it earlier"). The push plate's limits
    are in :func:`depths`.
    """
    back = A.TRAY_BACK_DEPTH - back_margin
    body = _jig_front_depth() - front_margin - A.BODY_FZ_FRONT + A.PLATE_FZ_TIP
    return min(back, body)


def insert_pose(slot: Slot, back_margin: float = BACK_MARGIN,
                front_margin: float = FRONT_MARGIN) -> np.ndarray:
    """Fully inserted, the tip at :func:`insert_tip_depth` (D28, no push plate)."""
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


def grasp_sequence(process: dict, slot: Slot, near_rise: float = 0.0, tool: Optional[dict] = None) -> dict:
    """The pick's flange poses by phase (:func:`pick_phases`).

    Without a push plate these are cell_core's seven grasp poses exactly; with
    one, ``insert`` stops short of the tray and a ``push`` phase follows.
    Standoffs, the carry tilt and the push-plate parameters come from *process*;
    *tool* is the gripper geometry read from the robot's URDF
    (:func:`tool.tool_geometry`).
    """
    return {p.name: p.T for p in pick_phases(process, slot, tool, near_rise)}


def release_sequence(process: dict, slot: Slot, near_rise: float = 0.0, tool: Optional[dict] = None) -> dict:
    """The place's flange poses by phase (:func:`place_phases`).

    Without a push plate these are the grasp poses reversed (cell_core): the tray
    goes down where it was picked up. With one, it goes down ``place_offset``
    shallower, so they differ.
    """
    return {p.name: p.T for p in place_phases(process, slot, tool)}


# ─────────────────────────────────────────────────────────────────────────────
# Pick and place with the push plate (user 2026-11-12)
# ─────────────────────────────────────────────────────────────────────────────
# The push plate does not move. On pick the arm stops short of the tray, then
# advances with the jaws open so the plate pushes the tray back on its columns:
# the tray ends registered against the plate, which is what tilt could not do.
# On place the tray is set down slightly shallower than its seat, so the next
# pick has room to push. Parameters, ``process.push_plate``:
#
#   margin        insert stops this far before the plate meets the tray (2 mm)
#   overtravel    the push ends this far past the tray's nominal seat (D30: 2 mm)
#   place_offset  a placed tray rests this far shallower than its seat (D30: 2 mm)
#
# "Depth" is behind the rack face along the approach axis.

#: Deepest a tray may be pushed past its seat before its rails leave the front
#: columns (rails rest on the column tops, D17).
def max_push_past_seat() -> float:
    col_front_inner = min(c[1] - s[1] / 2.0 for part, (c, s) in A.JIG_BOXES.items()
                          if part.startswith("col") and c[1] > 0)
    return A.TRAY_Y / 2.0 - col_front_inner


@dataclass(frozen=True)
class Phase:
    """One step of a pick or place: the flange pose and where the target tray is.

    ``tray`` is ``"seated"`` (on its jig, ``shift`` metres toward the robot from
    its nominal seat) or ``"carried"`` (on the plate, its centre at flange z
    ``carried_z``). ``contact`` marks a phase in which the push plate touches the
    tray on purpose.
    """
    name: str
    T: np.ndarray
    tray: str
    shift: float = 0.0
    carried_z: float = 0.0
    contact: bool = False
    #: Poses from the previous phase to this one (both included) when the move
    #: between them is not a free one and must be checked along the way.
    path: tuple = ()
    #: Clearance every obstacle must keep along ``path`` (None: the phase's own rules).
    path_clearance: Optional[float] = None


def _push_params(process: dict, tool: Optional[dict]):
    if not tool or tool.get("push_front_z") is None:
        return None
    pp = process.get("push_plate") or {}
    return {"face": float(tool["push_front_z"]), "margin": float(pp.get("margin", 0.0)),
            "overtravel": float(pp.get("overtravel", 0.0)), "place_offset": float(pp.get("place_offset", 0.0))}


def depths(process: dict, tool: Optional[dict]) -> dict:
    """Plate-tip depths of the pick and place: ``insert``, ``push``, ``place``.

    Without a push plate the D28 sequence: one depth, no push.
    """
    d28 = insert_tip_depth()
    pp = _push_params(process, tool)
    if pp is None:
        return {"insert": d28, "push": None, "place": d28}
    K = A.PLATE_FZ_TIP - pp["face"]          # tip ahead of the push face
    f0 = A.TRAY_FRONT_DEPTH                  # nominal tray front
    return {"insert": min(d28, f0 - pp["place_offset"] - pp["margin"] + K),
            "push": min(d28, f0 + pp["overtravel"] + K),
            "place": min(d28, f0 - pp["place_offset"] + K)}


def _at_depth(slot: Slot, depth: float, lift: float = 0.0) -> np.ndarray:
    tip_dist = slot.label_plane_dist + A.LABEL_PROUD_OF_FACE + depth
    return _pose(slot, tip_dist - A.PLATE_FZ_TIP, insertion_flange_z(slot) + lift)


@dataclass(frozen=True)
class TiltMotion:
    """The withdraw -> carry move as one motion (user 2026-11-12): the plate tilts
    by the carry tilt about a pivot while the flange backs out ``retreat``.

    The pivot is a point of the flange's y-z plane (flange frame at withdraw);
    the nearer it is to the tray's back end, the less that end rises and the
    more the flange drops. Rotation and retreat run together, linearly in the
    motion parameter, so the tray goes down and back while it tilts.
    """
    pivot_y: float = 0.0
    pivot_z: float = 0.0
    retreat: float = 0.0
    samples: int = 16
    #: Clearance to every obstacle along the motion (user 2026-11-13: >= 10 mm).
    clearance: Optional[float] = None

    def pose(self, T_withdraw, tilt_rad: float, u: float) -> np.ndarray:
        P = F.make_pose(np.eye(3), (0.0, self.pivot_y, self.pivot_z))
        R = F.make_pose(F.rot_x(u * tilt_rad), (0.0, 0.0, 0.0))     # flange_rotation's tilt convention
        back = F.make_pose(np.eye(3), (0.0, 0.0, -u * self.retreat))
        return np.asarray(T_withdraw, dtype=float) @ back @ P @ R @ np.linalg.inv(P)

    def path(self, T_withdraw, tilt_rad: float) -> tuple:
        return tuple(self.pose(T_withdraw, tilt_rad, u) for u in np.linspace(0.0, 1.0, self.samples + 1))

    def displacement(self, tilt_rad: float) -> float:
        """How far the flange moves (m), withdraw -> carry."""
        return float(np.linalg.norm(self.pose(np.eye(4), tilt_rad, 1.0)[:3, 3]))


def tilt_motion(process: dict) -> TiltMotion:
    """``process.tilt_motion``: ``{pivot: [y, z], retreat: r, samples: n, clearance: c}``."""
    tm = process.get("tilt_motion") or {}
    py, pz = tm.get("pivot", (0.0, 0.0))
    c = tm.get("clearance")
    return TiltMotion(float(py), float(pz), float(tm.get("retreat", 0.0)), int(tm.get("samples", 16)),
                      None if c is None else float(c))


def pick_phases(process: dict, slot: Slot, tool: Optional[dict] = None, near_rise: float = 0.0) -> list:
    """The pick, phase by phase; the D28 grasp when the tool has no push plate."""
    pp = _push_params(process, tool)
    d = depths(process, tool)
    tilt = process.get("carry_tilt_deg", 0.0)
    seated = pp["place_offset"] if pp else 0.0           # worst case: a tray we placed shallow
    if pp:
        carried_z = pp["face"] + A.TRAY_Y / 2.0          # registered against the plate
    else:
        carried_z = A.PLATE_FZ_TIP + A.JIG_Y_OFFSET - d["insert"]
    out = [Phase("sight_far", sighting_pose(slot, process["standoff_far"]), "seated", seated),
           Phase("sight_near", sighting_pose(slot, process["standoff_near"], near_rise), "seated", seated),
           Phase("pre_insert", pre_insert_pose(slot), "seated", seated),
           Phase("insert", _at_depth(slot, d["insert"]), "seated", seated)]
    top = d["insert"]
    if pp:
        out.append(Phase("push", _at_depth(slot, d["push"]), "seated",
                         -(d["push"] - (A.PLATE_FZ_TIP - pp["face"]) - A.TRAY_FRONT_DEPTH), contact=True))
        top = d["push"]
    out += [Phase("lift", _at_depth(slot, top, LIFT_HEIGHT), "carried", carried_z=carried_z),
            Phase("withdraw", withdraw_pose(slot), "carried", carried_z=carried_z)]
    tm = tilt_motion(process)
    path = tm.path(withdraw_pose(slot), math.radians(tilt)) if tilt else ()
    out.append(Phase("carry", path[-1] if path else carry_pose(slot, tilt), "carried", carried_z=carried_z,
                     path=path, path_clearance=tm.clearance if path else None))
    return out


def place_phases(process: dict, slot: Slot, tool: Optional[dict] = None) -> list:
    """The place, phase by phase: arrive carried, set down, retreat."""
    pp = _push_params(process, tool)
    d = depths(process, tool)
    tilt = process.get("carry_tilt_deg", 0.0)
    if pp:
        carried_z = pp["face"] + A.TRAY_Y / 2.0
        placed = d["place"] - (A.PLATE_FZ_TIP - pp["face"])     # tray front depth when set down
        shift = A.TRAY_FRONT_DEPTH - placed                    # > 0: shallower than its seat
    else:
        carried_z = A.PLATE_FZ_TIP + A.JIG_Y_OFFSET - d["place"]
        shift = 0.0
    tm = tilt_motion(process)
    path = tm.path(withdraw_pose(slot), math.radians(tilt)) if tilt else ()
    return [Phase("carry", path[-1] if path else carry_pose(slot, tilt), "carried", carried_z=carried_z),
            Phase("withdraw", withdraw_pose(slot), "carried", carried_z=carried_z, path=tuple(reversed(path)),
                  path_clearance=tm.clearance if path else None),
            Phase("place_lift", _at_depth(slot, d["place"], LIFT_HEIGHT), "carried", carried_z=carried_z),
            Phase("place_down", _at_depth(slot, d["place"]), "seated", shift, contact=bool(pp)),
            Phase("retreat", pre_insert_pose(slot), "seated", shift)]

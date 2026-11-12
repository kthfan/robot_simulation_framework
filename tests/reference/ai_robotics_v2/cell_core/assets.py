"""Measured, non-negotiable primitives of the physical assets.

Everything here was extracted from the USD assets on the isaacsim host
(``tools/verify_flange_offsets.py``) or supplied directly by the user. These are
properties of the *hardware*, not of the cell layout — they do not vary between
deployments and are therefore not part of :mod:`cell_core.config`.

Nothing else in the project may hard-code these values, and no derived quantity
may be transcribed out of this module into a document or another module. An
earlier revision of the findings was corrupted precisely that way: a wrong
camera offset was copied into three documents and survived because each one
looked self-consistent.

Frame conventions
-----------------
World: Z up, the robot rotation axis at x=y=0, floor at z=0.

Flange (verified against ``tm5s.usd`` at the zero configuration):

===== ==================================================
``+X`` gripper open/close (lateral)
``+Y`` "up"; the bottom plate hangs at ``-Y``, the EIH camera sits at ``+Y``
``+Z`` approach / insertion axis
===== ==================================================

Jig: ``z=0`` at the jig origin (the top face of the shelf plate), ``+y`` points
out of the rack toward the robot, so the AprilGrid label plane sits at jig
``y = +0.157``.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Flange frame — tm5s.usd
# ─────────────────────────────────────────────────────────────────────────────
# EIH 5M camera -- the LOCATING camera. Flange -> optical centre from the
# hand-eye calibration (camera_param_raw/extrinsics.md, "EIH-Camera"); it looks
# along +Z, rotated ~180 deg about Z.
#
# D24: tm5s.usd did not model this camera. Its only camera was the D455 below,
# and D18 mistook that one for the EIH, moving every sighting pose, the
# plate-free standoff and the lower-layer clearance. findings/02 had the right
# numbers. The ``link_6/eih_camera`` prim added later sits 8 mm off in Z and
# has the wrong intrinsics; build_cell overrides it from these values (M4).
CAM_FX = -0.000197        #: camera optical centre, lateral
CAM_FY = 0.075155         #: camera optical centre above the flange axis
CAM_FZ = 0.044217         #: camera ahead of the flange along the approach axis
#: Flange -> optical-frame rotation, quaternion ``(x, y, z, w)``.
CAM_QUAT = (-0.0028787906838694576, -0.0016772167970156055,
            -0.9999942708314725, 0.0005981740931728217)

# RealSense D455 -- the SCANNING camera (M5), not used for locating. Measured
# from the rsd455_v1 prims in tm5s.usd; the calibrated D455 (+Y 0.1176,
# +Z 0.0949) agrees to ~12 mm, which confirms that both sources use this flange
# frame.
D455_FY = 0.11383
D455_FZ = 0.10670
D455_CASE_HALF_H = 0.0145  #: D455 case half-height
D455_CASE_HALF_W = 0.0619  #: D455 case half-width (flange x)
D455_CASE_DEPTH = 0.026    #: D455 case, back to front face (flange z)

# The real gripper, supplied by the user 2026-10-06 (D28). It supersedes the
# plate measured from tm5s.usd (tip 0.31849, root 0.10349, -0.04269..-0.03269):
# the sim asset hangs the plate 3.3 mm high and 2.5 mm short.
#: Gripper body front: the link6 mount plate spans 0-10 mm, the gripper 10-93 mm.
BODY_FZ_FRONT = 0.093
BODY_FX = 0.061           #: gripper body half-width (sideways)
PLATE_FZ_TIP = 0.321      #: plate tip, along the approach axis
PLATE_FZ_ROOT = 0.0
PLATE_FY_BOT = -0.046     #: plate underside
PLATE_FY_TOP = -0.036     #: plate upper face (the tray-bearing surface)
PLATE_FX = 0.03000        #: plate half-width (tm5s.usd)
#: Plate/tray friction (static, dynamic): the plate's material, combine mode
#: max, so the pair gets exactly these (user 2026-10-20, D30). At the asset's
#: Teflon 0.3/0.1 the carried tray crept 8 mm on the plate in the sim.
PLATE_FRICTION = (0.5, 0.4)

# Measured from tm5s.usd; not superseded.
BODY_FY_TOP = 0.03931     #: gripper body, topmost point
JAW_FZ_TIP = 0.28036
JAW_FY_TOP = 0.03956

#: Flange offset from the shoulder (joint_2) along the robot axis, from
#: tm5s_v4.urdf joint_1 origin. Used to express reach as a shoulder radius.
SHOULDER_OFFSET_Z = 0.1484

# ─────────────────────────────────────────────────────────────────────────────
# Jig / tray / label — tray_jig_with_label.usd, tray.usd
# ─────────────────────────────────────────────────────────────────────────────
JIG_BASE_TOP = 0.005      #: jig base plate spans jig z [0, 0.005]
JIG_COL_TOP = 0.035       #: columns span jig z [0.005, 0.035]
JIG_COL_X = 0.0655        #: column centres, jig x
JIG_COL_Y = 0.1250        #: column centres, jig y
JIG_POCKET_X = 0.106      #: clear width between columns
JIG_Y_OFFSET = 0.135      #: jig centre, behind the rack front face
JIG_X_HALF = 0.078        #: half-width of one jig along the rack width

TRAY_X = 0.1329           #: tray width
TRAY_Y = 0.2470           #: tray length (depth direction)
TRAY_Z = 0.0851           #: tray height
TRAY_RECESS = 0.00250     #: underside recess over the plate corridor (measured)
TRAY_RAIL_W = 0.0065      #: load-bearing rail width at each long edge
TRAY_MASS = 1.2           #: kg, user-supplied

# Jaws (D30). Both jaw joints open their jaw outward along flange x, so the gap
# between the jaws is 2 (JAW_X_IN + q). The faces are the jaw colliders of
# tm5s.usd at q = 0; the stroke and the fully-open gap are the real gripper's
# (user 2026-10-10), which fixes the joint range.
JAW_X_IN = 0.02565        #: jaw inner face at q = 0, flange |x|
JAW_X_OUT = 0.0310        #: jaw outer face at q = 0, flange |x|
JAW_STROKE = 0.051        #: travel per jaw
JAW_GAP_OPEN = 0.173      #: between the jaws, fully open
#: Width-fit close (D5): clearance left on each side of the tray.
JAW_FIT_CLEARANCE = 0.0005


def jaw_gap(q: float) -> float:
    """Clear width between the jaws at joint value *q*."""
    return 2.0 * (JAW_X_IN + q)


def jaw_q(gap: float) -> float:
    """Joint value that leaves *gap* between the jaws."""
    return gap / 2.0 - JAW_X_IN


JAW_Q_MAX = jaw_q(JAW_GAP_OPEN)
JAW_Q_MIN = JAW_Q_MAX - JAW_STROKE
JAW_Q_FIT = jaw_q(TRAY_X + 2.0 * JAW_FIT_CLEARANCE)

LABEL_JIG_Y = 0.157       #: label plane, jig y
LABEL_Z_TOP = 0.0         #: label spans jig z [-0.030, 0]
LABEL_Z_BOT = -0.030

#: Every solid of ``tray_jig_with_label.usd`` as ``part: (centre, size)`` in the
#: jig frame, copied verbatim from its Cube prims (2026-10-03, M3). All of them
#: are axis-aligned boxes, so the collision world can carry the jig exactly.
#: The constants above (``JIG_BASE_TOP``, ``JIG_COL_*``, ``JIG_X_HALF``) are
#: views of this table, and a test keeps them consistent with it.
JIG_BOXES = {
    "base":      ((0.0, 0.0, 0.0025), (0.156, 0.27, 0.005)),
    "col_pp":    ((0.0655, 0.125, 0.02), (0.025, 0.02, 0.03)),
    "col_pm":    ((0.0655, -0.125, 0.02), (0.025, 0.02, 0.03)),
    "col_mp":    ((-0.0655, 0.125, 0.02), (0.025, 0.02, 0.03)),
    "col_mm":    ((-0.0655, -0.125, 0.02), (0.025, 0.02, 0.03)),
    "label":     ((0.0, 0.15775, -0.015), (0.1125, 0.001, 0.03)),
    "grid":      ((0.0, 0.157, -0.015), (0.1125, 0.0005, 0.03)),
    "hang_p":    ((0.065, 0.1575, -0.0115), (0.018, 0.0015, 0.037)),
    "hang_m":    ((-0.065, 0.1575, -0.0115), (0.018, 0.0015, 0.037)),
    "arm_p":     ((0.065, 0.14625, 0.00625), (0.018, 0.021, 0.0015)),
    "arm_m":     ((-0.065, 0.14625, 0.00625), (0.018, 0.021, 0.0015)),
    "clip_p":    ((0.065, 0.13575, 0.0095), (0.018, 0.0015, 0.008)),
    "clip_m":    ((-0.065, 0.13575, 0.0095), (0.018, 0.0015, 0.008)),
}
TAGS_X = 4                #: AprilGrid layout, as the generator prints it
TAGS_Y = 1
TAG_SIZE = 0.020
TAG_SEPARATION = 0.0075
GRID_W = TAGS_X * TAG_SIZE + (TAGS_X - 1) * TAG_SEPARATION   #: AprilGrid extent
GRID_H = TAGS_Y * TAG_SIZE + (TAGS_Y - 1) * TAG_SEPARATION

# ─────────────────────────────────────────────────────────────────────────────
# Rack construction defaults
# ─────────────────────────────────────────────────────────────────────────────
# These are real-cell values (D16) and are *also* carried per-rack in the config,
# because the environment generator exposes them as parameters. The values here
# are the defaults used when no rack is in hand.
DEFAULT_PLANE_THICKNESS = 0.020
DEFAULT_CYLINDER_RADIUS = 0.020
DEFAULT_JIG_GAP = 0.016
DEFAULT_RACK_DEPTH = 0.308


def jaw_open_q(jig_gap: float = DEFAULT_JIG_GAP) -> float:
    """The open command for inserting (D30).

    The jaw tips reach down past the column tops, so going in they pass between
    the target jig's front columns and the neighbouring jig's. The jaw is
    centred in that corridor, capped at fully open.
    """
    c, s = JIG_BOXES["col_pp"]
    own = c[0] + s[0] / 2.0
    neighbour = 2.0 * JIG_X_HALF + jig_gap - own
    mid = (own + neighbour) / 2.0 - (JAW_X_IN + JAW_X_OUT) / 2.0
    return min(mid, JAW_Q_MAX)


JAW_Q_OPEN = jaw_open_q()

# ─────────────────────────────────────────────────────────────────────────────
# Sim builder constants — build_env/create_robot_arm_base.py, create_environment.py
# ─────────────────────────────────────────────────────────────────────────────
# Fixed in the builder rather than exposed by the generator, so no preset can
# change them. The collision world needs them; the measured-stage test pins them.
MOUNT_BASE_XY = 0.55      #: pedestal base cube footprint (its height is the config's base_size)
MOUNT_LEG_RADIUS = 0.02
MOUNT_LEG_GAP = 0.10      #: clear space between adjacent pedestal legs
TABLE_WIDTH = 1.0         #: table top along its lateral axis

# ─────────────────────────────────────────────────────────────────────────────
# EIH intrinsics — distance-dependent focal configs (D10)
# ─────────────────────────────────────────────────────────────────────────────
#: Known anchor points. Intrinsics MUST match the commanded focal config; using
#: the wrong set silently corrupts every pose estimate. Extend as the
#: sharpness/reprojection sweep produces more measurements. Values are the ROS
#: ``k`` of camera_param_raw/eih_camera_intrinsics.md, copied in full.
FOCAL_CONFIGS = {
    6: dict(fx=2680.3818359375, fy=2680.74462890625,
            cx=1231.44580078125, cy=976.5955810546875, at=0.35),
    2: dict(fx=2653.350341796875, fy=2653.388427734375,
            cx=1232.0093994140625, cy=979.5687866210938, at=0.12),
}
EIH_WIDTH = 2592           #: EIH image width, px
EIH_HEIGHT = 1944          #: EIH image height, px


def focal_config_for(distance: float) -> int:
    """Return the focal config whose calibration anchor is nearest *distance*.

    Only two anchors are measured, so this is a nearest-neighbour lookup, not an
    interpolation. Anything between them is an extrapolation and should be
    confirmed experimentally before being trusted.
    """
    return min(FOCAL_CONFIGS, key=lambda k: abs(FOCAL_CONFIGS[k]["at"] - distance))


# ─────────────────────────────────────────────────────────────────────────────
# Derived — flange frame
# ─────────────────────────────────────────────────────────────────────────────
#: Vertical drop from the camera optical centre to the plate underside. This is
#: the number that governs lower-layer insertion clearance.
CAM_TO_PLATE_BOT = CAM_FY - PLATE_FY_BOT
CAM_TO_PLATE_TOP = CAM_FY - PLATE_FY_TOP

#: How far the plate tip reaches beyond the camera along the approach axis. If
#: the camera standoff exceeds this, the plate never enters the shelf at all.
PLATE_AHEAD_OF_CAM = PLATE_FZ_TIP - CAM_FZ

PLATE_MID_FY = 0.5 * (PLATE_FY_BOT + PLATE_FY_TOP)
PLATE_THICK = PLATE_FY_TOP - PLATE_FY_BOT

# ─────────────────────────────────────────────────────────────────────────────
# Derived — jig frame
# ─────────────────────────────────────────────────────────────────────────────
TRAY_BOT = JIG_COL_TOP                    #: tray rails rest on the columns
TRAY_TOP = TRAY_BOT + TRAY_Z

#: Vertical corridor available to the plate: jig base plate -> recessed underside
CORRIDOR_BOT = JIG_BASE_TOP
CORRIDOR_TOP = TRAY_BOT + TRAY_RECESS
CORRIDOR_H = CORRIDOR_TOP - CORRIDOR_BOT
CORRIDOR_MID = 0.5 * (CORRIDOR_BOT + CORRIDOR_TOP)

#: Corridor height if the tray underside turns out to be flat. D17: design to
#: this, because a 2.5 mm feature may not survive CAD export to the real tray.
CORRIDOR_H_FLAT = JIG_COL_TOP - JIG_BASE_TOP

#: Tray front/back edges relative to the rack front face (positive = deeper).
TRAY_FRONT_DEPTH = JIG_Y_OFFSET - TRAY_Y / 2.0
TRAY_BACK_DEPTH = JIG_Y_OFFSET + TRAY_Y / 2.0

#: How far the label plane stands proud of the rack front face, toward the robot.
LABEL_PROUD_OF_FACE = LABEL_JIG_Y - JIG_Y_OFFSET
LABEL_MID_Z = 0.5 * (LABEL_Z_TOP + LABEL_Z_BOT)

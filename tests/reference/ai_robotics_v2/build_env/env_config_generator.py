"""Smart configuration generator for the ReKep Digital Twin environment.

Given high-level distance and count parameters, derives the full geometry
configuration dict expected by :func:`create_environment`.

Usage example::

    from tools.env_config_generator import generate_env_config
    from tools.create_environment import create_environment

    cfg = generate_env_config(
        dist_right_rack=1.185,
        dist_left_rack=1.185,
        dist_table=0.85,
        jig_gap=0.015,
        n_jigs_right=3,
        n_jigs_left=4,
        right_rack_planes=[0.55, 0.85],
        left_rack_planes=[0.25, 0.55, 0.85, 1.15],
        table_height=0.7,
        table_jig_x=0.0,
    )
    create_environment(stage, root_path="/World_0/Environment", **cfg)
"""

# Jig geometry constants shared with create_rack_shelf_with_tray_jig.py
_JIG_X_HALF = 0.078   # half-width of one tray-jig in local rack X direction
_CYLINDER_RADIUS = 0.02
_RACK_DEPTH = 0.36
_TABLE_SIZE_Y = 0.36  # must match hardcoded value in create_environment()


def _min_rack_width(n_jigs: int, jig_gap: float, cylinder_radius: float) -> float:
    """Return the minimum shelf X size (metres) needed to hold *n_jigs* columns."""
    if n_jigs <= 0:
        return 0.0
    return (
        2 * _JIG_X_HALF
        + 2 * cylinder_radius
        + jig_gap
        + (n_jigs - 1) * (jig_gap + 2 * _JIG_X_HALF)
    )


def generate_env_config(
    *,
    # ── Nearest distances (m) from robot origin to nearest rack/table face ──
    dist_right_rack: float = 1.185,
    dist_left_rack: float = 1.185,
    dist_table: float = 0.85,
    # ── Jig layout ───────────────────────────────────────────────────────────
    jig_gap: float = 0.015,
    n_jigs_right: int = 3,
    n_jigs_left: int = 4,
    right_rack_planes: "list[float] | None" = None,
    left_rack_planes: "list[float] | None" = None,
    # ── Rack dimensions ──────────────────────────────────────────────────────
    rack_depth: float = _RACK_DEPTH,
    right_rack_height: float = 1.0,
    left_rack_height: float = 1.2,
    size_x_margin: float = 0.05,
    cylinder_radius: float = _CYLINDER_RADIUS,
    # ── Table ────────────────────────────────────────────────────────────────
    table_height: float = 0.7,
    table_jig_x: float = 0.0,
    table_size_y: float = _TABLE_SIZE_Y,
    # ── AprilGrid IDs ────────────────────────────────────────────────────────
    start_id: int = 0,
    table_jig_id: "int | None" = None,
    # ── Robot ──────────────────────────────────────────────────────── 
    robot_leg_height: float = 0.5,
) -> dict:
    """Generate a complete :func:`create_environment` configuration dict.

    Parameters
    ----------
    dist_right_rack : float
        Nearest distance (m) from robot origin to the front face of the right rack.
        The front face is the side facing the robot (+X direction from robot).
    dist_left_rack : float
        Nearest distance (m) from robot origin to the front face of the left rack.
        The front face faces the robot (-X direction from robot).
    dist_table : float
        Nearest distance (m) from robot origin to the front edge of the table.
        The front edge faces the robot (+Y direction from robot).
    jig_gap : float
        Horizontal gap (m) between adjacent tray jigs (and before the first jig).
    n_jigs_right : int
        Number of tray jigs per shelf plane on the right rack.
    n_jigs_left : int
        Number of tray jigs per shelf plane on the left rack.
    right_rack_planes : list[float] | None
        Z heights (m) of shelf planes on the right rack.
        Default: ``[0.55, 0.85]``.
    left_rack_planes : list[float] | None
        Z heights (m) of shelf planes on the left rack.
        Default: ``[0.25, 0.55, 0.85, 1.15]``.
    rack_depth : float
        Depth (Y dimension, m) of both racks.
    right_rack_height : float
        Total height (m) of the right rack frame.
    left_rack_height : float
        Total height (m) of the left rack frame.
    size_x_margin : float
        Extra width (m) added beyond the minimum shelf width needed for the jigs.
        Must be less than ``jig_gap + 2 * JIG_X_HALF`` to avoid an extra column.
    cylinder_radius : float
        Radius (m) of the rack's vertical corner cylinders.
    table_height : float
        Height (m) of the table cuboid.
    table_jig_x : float
        X offset (m) of the tray jig on the table, in the table's local frame.
    table_size_y : float
        Depth (Y dimension, m) of the table.  Must match the value hardcoded in
        :func:`create_environment` (currently ``0.36``).
    start_id : int
        First AprilGrid grid-ID to assign.  IDs are assigned sequentially:
        right rack (bottom plane first, left-to-right), then left rack, then table.
    table_jig_id : int | None
        Explicit AprilGrid ID for the table jig.  If ``None``, the next ID after
        the rack jigs is used.
    robot_leg_height: float

    Returns
    -------
    dict
        Keyword arguments suitable for passing directly to :func:`create_environment`
        (excluding ``stage``, ``root_path``, ``arm_usd_path``, ``jig_usd_path``,
        and ``label_image_dir``).
    """
    if right_rack_planes is None:
        right_rack_planes = [0.55, 0.85]
    if left_rack_planes is None:
        left_rack_planes = [0.25, 0.55, 0.85, 1.15]

    # ── Rack shelf widths ──────────────────────────────────────────────────
    right_size_x = _min_rack_width(n_jigs_right, jig_gap, cylinder_radius) + size_x_margin
    left_size_x  = _min_rack_width(n_jigs_left,  jig_gap, cylinder_radius) + size_x_margin

    right_rack_size = (right_size_x, rack_depth, right_rack_height)
    left_rack_size  = (left_size_x,  rack_depth, left_rack_height)

    # ── Rack world-frame positions ─────────────────────────────────────────
    # Right rack: RotateZ(+90°) → front face at world_x = pos_x
    #             Center in world Y → pos_y = -right_size_x / 2
    right_rack_position = (dist_right_rack, -right_size_x / 2.0, 0.0)

    # Left rack: RotateZ(-90°) → front face at world_x = pos_x
    #            Center in world Y → pos_y = +left_size_x / 2
    left_rack_position = (-dist_left_rack, left_size_x / 2.0, 0.0)

    # ── Table position ─────────────────────────────────────────────────────
    # Table cube is centered: front edge at world_y = pos_y - table_size_y/2 = dist_table
    table_position = (0.0, dist_table + table_size_y / 2.0, 0)

    # ── AprilGrid ID assignment ────────────────────────────────────────────
    # Right rack: bottom plane first, then up; left-to-right within each plane
    current_id = start_id
    right_rack_ids = []
    for _ in right_rack_planes:
        row = list(range(current_id, current_id + n_jigs_right))
        right_rack_ids.append(row)
        current_id += n_jigs_right

    # Left rack: same ordering
    left_rack_ids = []
    for _ in left_rack_planes:
        row = list(range(current_id, current_id + n_jigs_left))
        left_rack_ids.append(row)
        current_id += n_jigs_left

    # Table jig
    if table_jig_id is None:
        table_jig_id = current_id

    # ── Exists matrices (all slots filled) ────────────────────────────────
    right_rack_jig_mask = [[True] * n_jigs_right for _ in right_rack_planes]
    left_rack_jig_mask  = [[True] * n_jigs_left  for _ in left_rack_planes]

    # ── Tray masks ────────────────────────────────────────────────────────
    # Right rack: all jigs have a tray; left rack: no trays
    right_rack_tray_mask = [[True] * n_jigs_right for _ in right_rack_planes]
    left_rack_tray_mask  = [[False]  * n_jigs_left  for _ in left_rack_planes]

    return dict(
        table_height=table_height,
        table_position=table_position,
        table_jig_x=table_jig_x,
        table_jig_id=table_jig_id,
        right_rack_position=right_rack_position,
        right_rack_size=right_rack_size,
        right_rack_planes=right_rack_planes,
        right_rack_ids=right_rack_ids,
        right_rack_jig_mask=right_rack_jig_mask,
        right_rack_tray_mask=right_rack_tray_mask,
        left_rack_position=left_rack_position,
        left_rack_size=left_rack_size,
        left_rack_planes=left_rack_planes,
        left_rack_ids=left_rack_ids,
        left_rack_jig_mask=left_rack_jig_mask,
        left_rack_tray_mask=left_rack_tray_mask,
        jig_gap=jig_gap,
        robot_leg_height=robot_leg_height,
    )

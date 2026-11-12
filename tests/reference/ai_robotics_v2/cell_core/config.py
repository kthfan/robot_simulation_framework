"""Cell configuration: schema, validation, and shipped presets.

Separation of concerns
----------------------
:mod:`cell_core.assets` holds what the *hardware* is ??gripper, tray, jig and
camera dimensions measured from the USD assets. This module holds what the
*installation* is ??where the racks stand, how tall the robot is mounted, how
many jigs per shelf. Assets do not vary between deployments; configuration does.

Presets
-------
``as_built``
    The real cell exactly as measured (D16). The robot sits at
    ``origin_z = 0.6765``. **This preset is a record and must not be edited to
    track design changes** ??it is what the physical cell is today.

``recommended``
    D19: the same cell with the mounting leg extended by 100 mm, which is what
    the reachability sweep says is needed to service every slot. It differs from
    ``as_built`` in exactly one number, and a test enforces that.

Anything else is configuration, not a code change: build a :class:`CellConfig`
or edit a YAML file.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import assets as A

PRESET_DIR = Path(__file__).resolve().parent / "presets"

#: Mounting height below which the sweep cannot service every slot (findings/04
#: §3.1: the cliff sits between 0.7315 and 0.7340 at the measured rack distance).
ORIGIN_Z_THRESHOLD = 0.733

#: Largest carry tilt that could plausibly be intended, in degrees.
#:
#: Not a measured limit. It exists to catch a radians value pasted into a
#: degrees field, and a tilt so steep the tray would slide off the back rather
#: than settle against it. The real ceiling comes from friction, once measured.
MAX_CARRY_TILT_DEG = 30.0


class CellConfigError(ValueError):
    """Raised for a configuration that cannot describe a buildable cell."""


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RobotMount:
    """The pedestal the arm stands on.

    Expressed as the physical stack rather than as a single height, so that
    changing the build means changing the part that actually changes.
    """

    base_size: float          #: arm base cube, floor-plate to arm origin
    leg_height: float         #: pedestal leg
    plate_thickness: float    #: mounting plate between leg and base
    #: Turn of the arm about its first joint axis, world from base, in degrees.
    #: The cell stands the arm half a turn round (the stage builder's RotateZ
    #: 180): its base +x faces world -x. A default rather than a preset key, so
    #: the presets, which are records, stay as they were.
    base_yaw_deg: float = 180.0

    @property
    def origin_z(self) -> float:
        """World height of the arm's first joint origin."""
        return self.base_size + self.leg_height + self.plate_thickness

    @property
    def base_pose(self) -> np.ndarray:
        """4x4 world pose of the arm's base frame."""
        c, s = math.cos(math.radians(self.base_yaw_deg)), math.sin(math.radians(self.base_yaw_deg))
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[2, 3] = self.origin_z
        return T


@dataclass(frozen=True)
class RackConfig:
    name: str
    role: str                              #: "incoming" | "outgoing"
    approach: tuple                        #: unit horizontal vector, robot -> rack
    face_distance: float                   #: robot axis -> rack front face
    width: float                           #: along the rack face
    depth: float
    layer_heights: tuple                   #: world z of each shelf plane, ascending
    n_jigs: int                            #: jig columns per shelf
    plane_thickness: float = A.DEFAULT_PLANE_THICKNESS
    cylinder_radius: float = A.DEFAULT_CYLINDER_RADIUS
    jig_gap: float = A.DEFAULT_JIG_GAP

    @property
    def label_plane_distance(self) -> float:
        """Robot axis -> AprilGrid plane; the label stands proud of the face."""
        return self.face_distance - A.LABEL_PROUD_OF_FACE


@dataclass(frozen=True)
class TableConfig:
    """The inspection table.

    ``confirmed`` is deliberately part of the schema: these values are still the
    environment generator's defaults and have not been measured against the real
    cell. Consumers that care about accuracy should check it.
    """

    face_distance: float
    height: float
    depth: float
    jig_x: float = 0.0
    approach: tuple = (0.0, 1.0, 0.0)
    plane_thickness: float = A.DEFAULT_PLANE_THICKNESS
    confirmed: bool = False

    @property
    def label_plane_distance(self) -> float:
        return self.face_distance - A.LABEL_PROUD_OF_FACE


@dataclass(frozen=True)
class CellConfig:
    name: str
    description: str
    robot: RobotMount
    incoming: RackConfig
    outgoing: RackConfig
    table: TableConfig
    standoff_far: float
    standoff_near: float
    #: How far the plate tips back while carrying a tray, in **degrees**.
    #:
    #: The unit is in the name on purpose. Every other angle in this codebase is
    #: radians, and this one is human-edited in YAML, so the suffix is the only
    #: thing standing between a reader and a silent factor-of-57 error.
    #:
    #: Defaults to zero, meaning no tilt. The angle that actually retains a tray
    #: depends on the friction coefficient of the plate/tray pair, which nobody
    #: has measured. Rather than invent one, the geometry is built to accept any
    #: angle and the number waits for a measurement.
    carry_tilt_deg: float = 0.0

    @property
    def racks(self) -> tuple:
        return (self.incoming, self.outgoing)

    @property
    def carry_tilt(self) -> float:
        """The carry tilt in radians, ready for :mod:`cell_core.frames`."""
        return math.radians(self.carry_tilt_deg)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────
def to_dict(cfg: CellConfig) -> dict:
    """Plain nested dict, suitable for YAML. Derived values are not emitted."""
    return dataclasses.asdict(cfg)


def _build(cls, data: dict):
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise CellConfigError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = tuple(value) if isinstance(value, list) else value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise CellConfigError(f"{cls.__name__}: {exc}") from exc


def from_dict(data: dict) -> CellConfig:
    data = dict(data)
    try:
        robot = _build(RobotMount, data.pop("robot"))
        incoming = _build(RackConfig, data.pop("incoming"))
        outgoing = _build(RackConfig, data.pop("outgoing"))
        table = _build(TableConfig, data.pop("table"))
    except KeyError as exc:
        raise CellConfigError(f"missing section: {exc}") from exc
    return _build(
        CellConfig,
        dict(data, robot=robot, incoming=incoming, outgoing=outgoing, table=table),
    )


def load_config(path) -> CellConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh))


def save_config(cfg: CellConfig, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False, default_flow_style=False)


def available_presets() -> list:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_preset(name: str) -> CellConfig:
    path = PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        raise CellConfigError(
            f"unknown preset {name!r}; available: {', '.join(available_presets())}")
    return load_config(path)


def replace_mount(cfg: CellConfig, **changes) -> CellConfig:
    """Return a copy with the mounting stack altered. The original is untouched."""
    return dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, **changes))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter overrides, for scans
# ─────────────────────────────────────────────────────────────────────────────
def _set_origin_z(cfg: CellConfig, value: float) -> CellConfig:
    """Raise or lower the arm by changing the leg, the only part that can change.

    ``base_size`` and ``plate_thickness`` are hardware. Expressing a height scan
    as a leg length keeps the scan honest: if a height needs a negative leg, it
    is not a height you can build.
    """
    fixed = cfg.robot.base_size + cfg.robot.plate_thickness
    leg = value - fixed
    if leg < 0:
        raise CellConfigError(
            f"origin_z {value:.4f} needs a leg of {leg:.4f} m; the base and "
            f"plate alone are already {fixed:.4f} m")
    return replace_mount(cfg, leg_height=leg)


def _set_both_racks(field: str):
    def setter(cfg: CellConfig, value) -> CellConfig:
        return dataclasses.replace(
            cfg,
            incoming=dataclasses.replace(cfg.incoming, **{field: value}),
            outgoing=dataclasses.replace(cfg.outgoing, **{field: value}),
        )
    return setter


def _set_table(field: str):
    def setter(cfg: CellConfig, value) -> CellConfig:
        return dataclasses.replace(
            cfg, table=dataclasses.replace(cfg.table, **{field: value}))
    return setter


def _set_top(field: str):
    def setter(cfg: CellConfig, value) -> CellConfig:
        return dataclasses.replace(cfg, **{field: value})
    return setter


#: Parameters a sweep may vary, mapped to the change they actually make.
#: A registry rather than attribute-path parsing: every scannable quantity is
#: one someone deliberately made scannable, and the failure mode for a typo is a
#: clear error instead of a silently misapplied value.
SCAN_PARAMS = {
    "origin_z": _set_origin_z,
    "rack_distance": _set_both_racks("face_distance"),
    "jig_gap": _set_both_racks("jig_gap"),
    "standoff_near": _set_top("standoff_near"),
    "standoff_far": _set_top("standoff_far"),
    "carry_tilt_deg": _set_top("carry_tilt_deg"),
    "table_height": _set_table("height"),
    "table_distance": _set_table("face_distance"),
}


def apply_override(cfg: CellConfig, name: str, value) -> CellConfig:
    """Return a copy of *cfg* with one named parameter changed."""
    if name not in SCAN_PARAMS:
        raise CellConfigError(
            f"unknown parameter {name!r}; available: {', '.join(sorted(SCAN_PARAMS))}")
    return SCAN_PARAMS[name](cfg, value)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def min_rack_width(n_jigs: int,
                   jig_gap: float = A.DEFAULT_JIG_GAP,
                   cylinder_radius: float = A.DEFAULT_CYLINDER_RADIUS) -> float:
    """Minimum rack width for *n_jigs* columns.

    Mirrors ``_min_rack_width`` in ``build_env/env_config_generator.py``; if that
    changes, this must change with it or racks will be generated too narrow.
    """
    if n_jigs <= 0:
        return 0.0
    return (2 * cylinder_radius + jig_gap + 2 * A.JIG_X_HALF
            + (n_jigs - 1) * (jig_gap + 2 * A.JIG_X_HALF))


#: Minimum vertical space a shelf layer needs: the tray sitting on its jig, plus
#: room for the plate to be withdrawn from underneath the one above.
MIN_LAYER_GAP = A.TRAY_TOP + A.CORRIDOR_H_FLAT


def _validate_rack(rack: RackConfig) -> None:
    where = f"rack {rack.name!r}"
    if rack.n_jigs <= 0:
        raise CellConfigError(f"{where}: n_jigs must be positive")
    if rack.face_distance <= 0:
        raise CellConfigError(f"{where}: face_distance must be positive")
    if not rack.layer_heights:
        raise CellConfigError(f"{where}: no layer_heights")

    heights = list(rack.layer_heights)
    if heights != sorted(heights):
        raise CellConfigError(f"{where}: layer_heights must be ascending")
    if any(h <= 0 for h in heights):
        raise CellConfigError(f"{where}: layer_heights must be above the floor")

    gaps = [b - a for a, b in zip(heights, heights[1:])]
    if any(g < MIN_LAYER_GAP for g in gaps):
        raise CellConfigError(
            f"{where}: layer gap {min(gaps):.3f} is below the {MIN_LAYER_GAP:.3f} "
            "needed to stack a tray and withdraw the plate")

    need = min_rack_width(rack.n_jigs, rack.jig_gap, rack.cylinder_radius)
    if need > rack.width + 1e-9:
        raise CellConfigError(
            f"{where}: {rack.n_jigs} jigs need {need:.4f} m of width "
            f"but the rack is {rack.width:.4f} m")


def validate(cfg: CellConfig) -> CellConfig:
    """Raise :class:`CellConfigError` if the cell could not be built or served.

    Returns the config so it can be used inline.
    """
    if cfg.robot.origin_z <= 0:
        raise CellConfigError("robot origin_z must be positive")
    for rack in cfg.racks:
        _validate_rack(rack)

    if cfg.standoff_far <= 0 or cfg.standoff_near <= 0:
        raise CellConfigError("standoffs must be positive")
    if cfg.standoff_near > cfg.standoff_far:
        raise CellConfigError("standoff_near must not exceed standoff_far")
    if cfg.standoff_near < A.PLATE_AHEAD_OF_CAM - 1e-9:
        # D18: below this the plate is inside the shelf during the close read and
        # fouls the tray on the layer below.
        raise CellConfigError(
            f"standoff_near {cfg.standoff_near:.4f} puts the plate "
            f"{A.PLATE_AHEAD_OF_CAM - cfg.standoff_near:.4f} m into the shelf; "
            f"minimum is {A.PLATE_AHEAD_OF_CAM:.4f}")

    # A negative tilt drops the plate tip, which sheds the tray instead of
    # retaining it. The upper bound is not a physical limit so much as a sanity
    # one: past MAX_CARRY_TILT_DEG the tray is closer to being tipped off the
    # back than held, and the number is far more likely to be a units mistake
    # than an intention.
    if not 0.0 <= cfg.carry_tilt_deg <= MAX_CARRY_TILT_DEG:
        raise CellConfigError(
            f"carry tilt {cfg.carry_tilt_deg:.2f} deg is outside "
            f"0..{MAX_CARRY_TILT_DEG:.0f} deg; the value is in degrees")
    return cfg

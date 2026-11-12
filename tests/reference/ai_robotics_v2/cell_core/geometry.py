"""Derived cell geometry: slots, clearances, and the tolerance budget.

Nothing here is a constant. Every value is computed from
:mod:`cell_core.assets` (what the hardware is) and a :class:`~cell_core.config.CellConfig`
(what the installation is), so that a change to either propagates instead of
silently disagreeing with a number copied elsewhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import assets as A
from .config import CellConfig, RackConfig, min_rack_width


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance
# ─────────────────────────────────────────────────────────────────────────────
def tolerance_budget(engagement: float = 0.225, use_recess: bool = True) -> dict:
    """Plate-vs-pocket clearance and the pose error it permits.

    *engagement* is how far the plate is inside the pocket when the angular
    error matters; *use_recess* selects the measured 2.5 mm tray underside
    recess. D17: design to ``use_recess=False``, because a feature that small
    may not survive CAD export to the physical tray. The recessed figures are
    headroom, not budget.
    """
    corridor = A.CORRIDOR_H if use_recess else A.CORRIDOR_H_FLAT
    v_slack = corridor - A.PLATE_THICK
    lat_slack = A.JIG_POCKET_X - 2 * A.PLATE_FX
    return {
        "corridor_h": corridor,
        "vertical_slack": v_slack,
        "vertical_half": v_slack / 2.0,
        "lateral_slack": lat_slack,
        "lateral_half": lat_slack / 2.0,
        "pitch_deg": math.degrees(math.asin(min(1.0, (v_slack / 2.0) / engagement))),
        "yaw_deg": math.degrees(math.asin(min(1.0, (lat_slack / 2.0) / engagement))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rack layout
# ─────────────────────────────────────────────────────────────────────────────
def jig_lateral_offsets(n_jigs: int, jig_gap: float = A.DEFAULT_JIG_GAP) -> list:
    """Lateral offsets of jig centres, symmetric about the rack centreline."""
    step = jig_gap + 2 * A.JIG_X_HALF
    return [(i - (n_jigs - 1) / 2.0) * step for i in range(n_jigs)]


def rack_fits(rack: RackConfig) -> bool:
    """Whether the rack is wide enough for the jig columns it declares."""
    need = min_rack_width(rack.n_jigs, rack.jig_gap, rack.cylinder_radius)
    return need <= rack.width + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Slots
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Slot:
    """One jig position: a rack, a shelf layer, and a column."""

    rack: str
    role: str
    layer: int
    column: int
    height: float             #: shelf plane centre, world z
    lateral: float            #: offset along the rack width
    approach: tuple           #: unit horizontal vector, robot -> into the slot
    label_plane_dist: float   #: robot axis -> AprilGrid plane
    plane_thickness: float

    @property
    def key(self) -> str:
        return f"{self.rack}/L{self.layer}/C{self.column}"

    @property
    def approach_vec(self) -> np.ndarray:
        return np.asarray(self.approach, dtype=float)

    @property
    def jig_origin_z(self) -> float:
        """Top face of the shelf plate, which is the jig's z origin."""
        return self.height + self.plane_thickness / 2.0

    @property
    def label_centre_z(self) -> float:
        return self.jig_origin_z + A.LABEL_MID_Z

    @property
    def tray_top_z(self) -> float:
        return self.jig_origin_z + A.TRAY_TOP


def enumerate_slots(cfg: CellConfig) -> list:
    """Every serviceable jig position in the cell, racks first, table last."""
    slots = []
    for rack in cfg.racks:
        offsets = jig_lateral_offsets(rack.n_jigs, rack.jig_gap)
        for li, height in enumerate(rack.layer_heights):
            for ci, lateral in enumerate(offsets):
                slots.append(Slot(
                    rack=rack.name, role=rack.role, layer=li, column=ci,
                    height=height, lateral=lateral, approach=tuple(rack.approach),
                    label_plane_dist=rack.label_plane_distance,
                    plane_thickness=rack.plane_thickness,
                ))
    t = cfg.table
    # Slot.height is a plate centre; the table's height is its top surface.
    slots.append(Slot(
        rack="table", role="inspection", layer=0, column=0,
        height=t.height - t.plane_thickness / 2.0, lateral=t.jig_x,
        approach=tuple(t.approach),
        label_plane_dist=t.label_plane_distance,
        plane_thickness=t.plane_thickness,
    ))
    return slots


#: The order the environment generator issues AprilGrid ids in.
_GRID_ID_ORDER = ("incoming", "outgoing", "inspection")


def grid_ids(cfg: CellConfig, start_id: int = 0) -> dict:
    """AprilGrid grid-id on each slot's label, keyed by :attr:`Slot.key`.

    The generator's rule: incoming rack, then outgoing, then the table; bottom
    shelf first; column 0 first. Marker ids are ``4 * grid_id + 0..3``.
    """
    slots = sorted(enumerate_slots(cfg),
                   key=lambda s: (_GRID_ID_ORDER.index(s.role), s.layer, s.column))
    return {s.key: start_id + i for i, s in enumerate(slots)}


# ─────────────────────────────────────────────────────────────────────────────
# Robot mount
# ─────────────────────────────────────────────────────────────────────────────
def shoulder_z(cfg: CellConfig) -> float:
    """World height of joint_2, the centre of the reach envelope."""
    return cfg.robot.origin_z + A.SHOULDER_OFFSET_Z


def shoulder_radius(cfg: CellConfig, position) -> float:
    """Distance from the shoulder to a world point.

    Useful as a cheap reach proxy, but only a proxy: the true envelope is not a
    sphere, because the wrist offsets make it lumpy. An analytic IK verdict
    (``tools/reach_sweep.py``) is authoritative.
    """
    p = np.asarray(position, dtype=float)
    return float(np.linalg.norm(p - np.array([0.0, 0.0, shoulder_z(cfg)])))


# ─────────────────────────────────────────────────────────────────────────────
# Clearance
# ─────────────────────────────────────────────────────────────────────────────
def lower_layer_clearance(layer_gap: float, rise: float = 0.0) -> float:
    """Plate underside versus the tray one layer down, during a close read.

    Positive is clear. The shelf plate thickness cancels, so this depends only
    on the layer gap and how far the camera is raised above the label centre.

    D24 (supersedes D18's -41.6 mm, which used the D455's position): at the
    real 0.25 m gap this is -2.9 mm with no rise.
    """
    # Both terms are measured from the same shelf plane, so the plate thickness
    # that positions the jig origin cancels and does not appear.
    plate_bottom = A.LABEL_MID_Z + rise - A.CAM_TO_PLATE_BOT
    tray_top_below = A.TRAY_TOP - layer_gap
    return plate_bottom - tray_top_below


def plate_free_standoff() -> float:
    """Smallest camera standoff at which the plate never enters the shelf."""
    return A.PLATE_AHEAD_OF_CAM


def plate_clears_lower_tray_standoff() -> float:
    """Smallest camera standoff at which the plate stops overhanging the tray below."""
    return A.PLATE_AHEAD_OF_CAM - (A.LABEL_PROUD_OF_FACE + A.TRAY_FRONT_DEPTH)


# ─────────────────────────────────────────────────────────────────────────────
# Optics
# ─────────────────────────────────────────────────────────────────────────────
def fov_half_height(standoff: float, config: int = None) -> float:
    """Half the vertical field of view, in metres, at *standoff*."""
    k = A.FOCAL_CONFIGS[config or A.focal_config_for(standoff)]
    return standoff * k["cy"] / k["fy"]


def fov_half_width(standoff: float, config: int = None) -> float:
    """Half the horizontal field of view, in metres, at *standoff*.

    The grid is wider than it is tall, so this is the dimension that limits how
    close the camera may read.
    """
    k = A.FOCAL_CONFIGS[config or A.focal_config_for(standoff)]
    return standoff * k["cx"] / k["fx"]


def grid_pixels(standoff: float, config: int = None) -> float:
    """Width of the AprilGrid in pixels at *standoff*.

    This is the accuracy currency: more pixels across the grid means a tighter
    pose estimate. It trades directly against clearance (see
    :func:`plate_free_standoff`).
    """
    k = A.FOCAL_CONFIGS[config or A.focal_config_for(standoff)]
    return A.GRID_W * k["fx"] / standoff

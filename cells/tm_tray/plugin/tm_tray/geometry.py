"""Slots and derived cell geometry (port of AI-Robotics-V2 ``cell_core.geometry``).

A :class:`Slot` is one jig position. In the framework it is carried as a
:class:`~rsf.model.world.Feature` of kind ``slot`` whose pose is the jig frame
(:func:`jig_pose`) and whose attributes are the slot's fields;
:func:`slot_of` turns the feature back into a :class:`Slot` for the TM-specific
geometry (grasp sequences, clearances) that is written against it.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from . import frames as F
from . import hardware as A


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance
# ─────────────────────────────────────────────────────────────────────────────
def tolerance_budget(engagement: float = 0.225, use_recess: bool = True) -> dict:
    """Plate-vs-pocket clearance and the pose error it permits (D17)."""
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
def min_rack_width(n_jigs: int, jig_gap: float = A.DEFAULT_JIG_GAP,
                   cylinder_radius: float = A.DEFAULT_CYLINDER_RADIUS) -> float:
    """Minimum rack width for *n_jigs* columns (legs at both ends, one gap)."""
    if n_jigs <= 0:
        return 0.0
    return (2 * cylinder_radius + jig_gap + 2 * A.JIG_X_HALF
            + (n_jigs - 1) * (jig_gap + 2 * A.JIG_X_HALF))


def jig_lateral_offsets(n_jigs: int, jig_gap: float = A.DEFAULT_JIG_GAP) -> list:
    """Lateral offsets of jig centres, symmetric about the rack centreline."""
    step = jig_gap + 2 * A.JIG_X_HALF
    return [(i - (n_jigs - 1) / 2.0) * step for i in range(n_jigs)]


#: Minimum vertical space a shelf layer needs: the tray on its jig, plus room to
#: withdraw the plate from under the one above.
MIN_LAYER_GAP = A.TRAY_TOP + A.CORRIDOR_H_FLAT


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


SLOT_FIELDS = tuple(f.name for f in dataclasses.fields(Slot))


def jig_pose(slot: Slot) -> np.ndarray:
    """World pose of the jig origin: jig +y faces the robot, +z is up."""
    a = slot.approach_vec / np.linalg.norm(slot.approach_vec)
    lat = F.lateral_axis(a)
    face = slot.label_plane_dist + A.LABEL_PROUD_OF_FACE
    centre = a * (face + A.JIG_Y_OFFSET) + lat * slot.lateral + F.UP * slot.jig_origin_z
    return F.make_pose(np.column_stack([lat, -a, F.UP]), centre)


def slot_attrs(slot: Slot) -> dict:
    d = dataclasses.asdict(slot)
    d["approach"] = [float(v) for v in slot.approach]
    return d


def slot_of(feature) -> Slot:
    """The :class:`Slot` a ``slot`` feature carries."""
    d = {k: feature.attrs[k] for k in SLOT_FIELDS}
    d["approach"] = tuple(d["approach"])
    return Slot(**d)


def slot_feature(slot: Slot, entity: str):
    from rsf.model.world import Feature
    return Feature(slot.key, jig_pose(slot), "slot", entity, slot_attrs(slot), frozenset({"slot", slot.role}))


def slots(world) -> list:
    """Every slot of *world*, in layout order (racks first, table last)."""
    return [slot_of(f) for f in world.features({"kind": "slot"})]


#: The order the AprilGrid ids are issued in.
GRID_ID_ORDER = ("incoming", "outgoing", "inspection")


def grid_ids(slot_list, start_id: int = 0) -> dict:
    """AprilGrid grid-id on each slot's label: incoming, outgoing, table; bottom
    shelf first; column 0 first. Marker ids are ``4 * grid_id + 0..3``."""
    ordered = sorted(slot_list, key=lambda s: (GRID_ID_ORDER.index(s.role), s.layer, s.column))
    return {s.key: start_id + i for i, s in enumerate(ordered)}


# ─────────────────────────────────────────────────────────────────────────────
# Clearance and optics
# ─────────────────────────────────────────────────────────────────────────────
def lower_layer_clearance(layer_gap: float, rise: float = 0.0) -> float:
    """Plate underside versus the tray one layer down, during a close read (D24)."""
    plate_bottom = A.LABEL_MID_Z + rise - A.CAM_TO_PLATE_BOT
    tray_top_below = A.TRAY_TOP - layer_gap
    return plate_bottom - tray_top_below


def plate_free_standoff() -> float:
    """Smallest camera standoff at which the plate never enters the shelf."""
    return A.PLATE_AHEAD_OF_CAM


def fov_half_height(standoff: float, config: int = None) -> float:
    k = A.FOCAL_CONFIGS[config or A.focal_config_for(standoff)]
    return standoff * k["cy"] / k["fy"]


def fov_half_width(standoff: float, config: int = None) -> float:
    k = A.FOCAL_CONFIGS[config or A.focal_config_for(standoff)]
    return standoff * k["cx"] / k["fx"]


def grid_pixels(standoff: float, config: int = None) -> float:
    k = A.FOCAL_CONFIGS[config or A.focal_config_for(standoff)]
    return A.GRID_W * k["fx"] / standoff

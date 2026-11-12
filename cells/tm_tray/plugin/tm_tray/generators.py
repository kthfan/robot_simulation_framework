"""Layout generators for the TM tray cell (port of AI-Robotics-V2 ``cell_core.scene``).

``tm_tray.pedestal``  the robot's mount stack; feature ``<name>/arm_base``
``tm_tray.rack``      a shelf rack: shelves, legs, one ``slot`` feature per jig position
``tm_tray.table``     the inspection table: top, one ``slot`` feature

Jigs and trays are not generated here: the layout places the jig asset at every
slot (``placements``) and the tray asset as an object (``objects``).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from rsf.model.registry import register
from rsf.model.world import Entity, Feature, Shape, Solid

from . import frames as F
from . import hardware as A
from .geometry import Slot, jig_lateral_offsets, slot_feature


class ParamError(ValueError):
    pass


def _params(cls, name: str, params: dict):
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(params) - fields
    if unknown:
        raise ParamError(f"{name}: unknown parameter(s) {sorted(unknown)}")
    try:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in params.items()})
    except TypeError as e:
        raise ParamError(f"{name}: {e}") from e


def _facing(approach) -> np.ndarray:
    """Box axes for something facing the robot: x along *approach*, y lateral, z up."""
    a = np.asarray(approach, dtype=float)
    a = a / np.linalg.norm(a)
    return np.column_stack([a, F.lateral_axis(a), F.UP])


def _box(name: str, entity: str, R, centre, size, *tags) -> Solid:
    return Solid(f"{entity}/{name}", F.make_pose(R, centre), Shape.box(size), entity, frozenset((entity,) + tags))


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PedestalParams:
    base_size: float          #: arm base cube, floor-plate to arm origin
    leg_height: float         #: pedestal leg
    plate_thickness: float    #: mounting plate between leg and base
    #: Turn of the arm about its first joint axis, world from base, degrees.
    #: The cell stands the arm half a turn round (D30).
    base_yaw_deg: float = 180.0
    base_xy: float = A.MOUNT_BASE_XY
    leg_radius: float = A.MOUNT_LEG_RADIUS
    leg_gap: float = A.MOUNT_LEG_GAP

    @property
    def origin_z(self) -> float:
        return self.base_size + self.leg_height + self.plate_thickness

    @property
    def base_pose(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.base_yaw_deg)), math.sin(math.radians(self.base_yaw_deg))
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[2, 3] = self.origin_z
        return T


@register("generator", "tm_tray.pedestal")
def pedestal(name: str, params: dict, ctx) -> Entity:
    m = _params(PedestalParams, name, params)
    I = np.eye(3)
    r = m.leg_radius
    span = m.leg_gap / 2.0 + r
    solids = [_box("base", name, I, (0.0, 0.0, m.base_size / 2.0), (m.base_xy, m.base_xy, m.base_size), "mount")]
    for i, (sx, sy) in enumerate(product((-1.0, 1.0), repeat=2)):
        solids.append(_box(f"leg{i}", name, I, (sx * span, sy * span, m.base_size + m.leg_height / 2.0),
                           (2 * r, 2 * r, m.leg_height), "mount"))
    plate_r = math.sqrt(2.0) * span + r
    solids.append(_box("plate", name, I, (0.0, 0.0, m.origin_z - m.plate_thickness / 2.0),
                       (2 * plate_r, 2 * plate_r, m.plate_thickness), "mount", "robot_seat"))
    arm = Feature(f"{name}/arm_base", m.base_pose, "mount", name, {"origin_z": m.origin_z})
    return Entity(name, "pedestal", tuple(solids), (arm,), {"params": dataclasses.asdict(m)})


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RackParams:
    role: str                              #: "incoming" | "outgoing" | anything a workflow selects on
    approach: tuple                        #: unit horizontal vector, robot -> rack
    face_distance: float                   #: robot axis -> rack front face
    width: float                           #: along the rack face
    depth: float
    layer_heights: tuple                   #: world z of each shelf plane centre, ascending
    n_jigs: int                            #: jig columns per shelf
    plane_thickness: float = A.DEFAULT_PLANE_THICKNESS
    cylinder_radius: float = A.DEFAULT_CYLINDER_RADIUS
    jig_gap: float = A.DEFAULT_JIG_GAP

    @property
    def label_plane_distance(self) -> float:
        return self.face_distance - A.LABEL_PROUD_OF_FACE


@register("generator", "tm_tray.rack")
def rack(name: str, params: dict, ctx) -> Entity:
    p = _params(RackParams, name, params)
    R = _facing(p.approach)
    a, lat = R[:, 0], R[:, 1]
    r, t, depth, width = p.cylinder_radius, p.plane_thickness, p.depth, p.width
    mid = p.face_distance + depth / 2.0
    solids = [_box(f"shelf{i}", name, R, a * mid + F.UP * h, (depth, width, t), "rack", "shelf")
              for i, h in enumerate(p.layer_heights)]
    height = max(p.layer_heights) + t
    legs = product((p.face_distance + r, p.face_distance + depth - r), (-(width / 2.0 - r), width / 2.0 - r))
    for i, (d, u) in enumerate(legs):
        solids.append(_box(f"leg{i}", name, R, a * d + lat * u + F.UP * (height / 2.0), (2 * r, 2 * r, height),
                           "rack", "leg"))
    features = []
    offsets = jig_lateral_offsets(p.n_jigs, p.jig_gap)
    for li, h in enumerate(p.layer_heights):
        for ci, lateral in enumerate(offsets):
            features.append(slot_feature(Slot(
                rack=name, role=p.role, layer=li, column=ci, height=h, lateral=lateral,
                approach=tuple(float(v) for v in p.approach), label_plane_dist=p.label_plane_distance,
                plane_thickness=t), name))
    return Entity(name, "rack", tuple(solids), tuple(features), {"params": dataclasses.asdict(p)})


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TableParams:
    face_distance: float
    height: float                          #: top surface, world z
    depth: float
    jig_x: float = 0.0
    approach: tuple = (0.0, 1.0, 0.0)
    plane_thickness: float = A.DEFAULT_PLANE_THICKNESS
    width: float = A.TABLE_WIDTH
    role: str = "inspection"
    #: The values are generator defaults, not measurements of the real cell.
    confirmed: bool = False

    @property
    def label_plane_distance(self) -> float:
        return self.face_distance - A.LABEL_PROUD_OF_FACE


@register("generator", "tm_tray.table")
def table(name: str, params: dict, ctx) -> Entity:
    t = _params(TableParams, name, params)
    R = _facing(t.approach)
    centre = R[:, 0] * (t.face_distance + t.depth / 2.0) + F.UP * (t.height / 2.0)
    top = _box("top", name, R, centre, (t.depth, t.width, t.height), "table")
    # Slot.height is a plate centre; the table's height is its top surface.
    slot = Slot(rack=name, role=t.role, layer=0, column=0, height=t.height - t.plane_thickness / 2.0,
                lateral=t.jig_x, approach=tuple(float(v) for v in t.approach),
                label_plane_dist=t.label_plane_distance, plane_thickness=t.plane_thickness)
    return Entity(name, "table", (top,), (slot_feature(slot, name),), {"params": dataclasses.asdict(t)})

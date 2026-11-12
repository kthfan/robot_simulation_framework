"""Audit checks for the TM tray cell (port of AI-Robotics-V2 ``cell_core.audit``).

Each check reads the world's generator parameters (entity ``meta["params"]``) and
``process`` values, and returns ``[]`` when the world has nothing it applies to.
Not ported: ``config_valid`` (a layout that fails its schema does not build),
``generator_fidelity`` (the Isaac stage is generated from the world in P5, so it
cannot differ from it).
"""
from __future__ import annotations

import math

from rsf.model.audit import SKIP, Check, check
from rsf.model.registry import register

from . import geometry as G
from . import hardware as A

#: Mounting height below which the reach sweep cannot service every slot
#: (findings/04 §3.1; D23 keeps it at 0.733).
ORIGIN_Z_THRESHOLD = 0.733
#: Largest plausible carry tilt, degrees (catches radians pasted into degrees).
MAX_CARRY_TILT_DEG = 30.0
#: Grid width floor in pixels; only catches an absurdly far camera.
MIN_GRID_PIXELS = 150.0
#: Smallest pitch tolerance worth calling a tolerance, degrees.
MIN_PITCH_DEG = 0.5


def _entities(world, kind):
    return [e for e in world.entities if e.kind == kind and "params" in e.meta]


@register("check", "tm_tray.rack")
def racks(world) -> list:
    out = []
    for e in _entities(world, "rack"):
        p = e.meta["params"]
        need = G.min_rack_width(p["n_jigs"], p["jig_gap"], p["cylinder_radius"])
        slack = p["width"] - need
        out.append(check(f"tm_tray.rack_width/{e.name}", slack >= -1e-9,
                         f"{p['n_jigs']} jigs need {need:.3f} m, shelf is {p['width']:.3f} m ({slack * 1000:.0f} mm spare)",
                         f"{p['n_jigs']} jigs need {need:.3f} m but the shelf is only {p['width']:.3f} m "
                         f"({-slack * 1000:.0f} mm short)", slack))
        h = list(p["layer_heights"])
        ok_order = h == sorted(h) and all(v > 0 for v in h)
        out.append(check(f"tm_tray.layer_order/{e.name}", ok_order, "layers ascend above the floor",
                         f"layer heights {h} must ascend and be above the floor"))
        gaps = [b - a for a, b in zip(h, h[1:])]
        if not gaps:
            out.append(Check(f"tm_tray.layer_gap/{e.name}", "pass", "single layer, no gap to check"))
            continue
        slack = min(gaps) - G.MIN_LAYER_GAP
        out.append(check(f"tm_tray.layer_gap/{e.name}", slack >= 0,
                         f"tightest layer gap {min(gaps):.3f} m, needs {G.MIN_LAYER_GAP:.3f} m ({slack * 1000:.0f} mm spare)",
                         f"tightest layer gap {min(gaps):.3f} m but a tray plus its corridor needs "
                         f"{G.MIN_LAYER_GAP:.3f} m ({-slack * 1000:.0f} mm short)", slack))
    return out


@register("check", "tm_tray.optics")
def optics(world) -> list:
    pr = world.process
    if "standoff_near" not in pr or "standoff_far" not in pr:
        return []
    near, far = pr["standoff_near"], pr["standoff_far"]
    out = [check("tm_tray.standoff_order", 0 < near <= far, f"standoffs {near} <= {far}",
                 f"standoffs must be positive with near <= far, got near {near}, far {far}")]
    need = G.plate_free_standoff()
    slack = near - need
    out.append(check("tm_tray.standoff_plate_free", slack >= -1e-9,
                     f"close read at {near:.4f} m keeps the plate outside the shelf "
                     f"(needs {need:.4f} m, {slack * 1000:.1f} mm spare)",
                     f"close read at {near:.4f} m puts the plate {-slack * 1000:.1f} mm inside the shelf; "
                     f"it must be at least {need:.4f} m", slack))
    for label, s in (("near", near), ("far", far)):
        half = G.fov_half_width(s)
        sl = half - A.GRID_W / 2.0
        out.append(check(f"tm_tray.grid_in_view/{label}", sl >= 0,
                         f"at {s:.4f} m the frame is {2 * half:.3f} m wide, grid {A.GRID_W:.3f} m",
                         f"at {s:.4f} m the grid is cropped by {-sl * 1000:.0f} mm each side", sl))
        px = G.grid_pixels(s)
        out.append(check(f"tm_tray.grid_pixels/{label}", px >= MIN_GRID_PIXELS,
                         f"grid spans {px:.0f} px at {s:.4f} m (floor {MIN_GRID_PIXELS:.0f} px)",
                         f"grid spans only {px:.0f} px at {s:.4f} m, below {MIN_GRID_PIXELS:.0f} px", px))
    return out


@register("check", "tm_tray.process")
def process(world) -> list:
    pr = world.process
    out = []
    if "carry_tilt_deg" in pr:
        t = pr["carry_tilt_deg"]
        out.append(check("tm_tray.carry_tilt", 0.0 <= t <= MAX_CARRY_TILT_DEG, f"carry tilt {t:.2f} deg",
                         f"carry tilt {t:.2f} deg is outside 0..{MAX_CARRY_TILT_DEG:.0f} deg; the value is in degrees"))
    if "jaw_operating_range" in pr:
        r = pr["jaw_operating_range"]
        fit, open_ = A.JAW_Q_FIT, A.JAW_Q_OPEN
        out.append(check("tm_tray.jaw_range", r["min"] <= fit <= open_ <= r["max"],
                         f"width-fit {fit:.4f} and open {open_:.4f} lie in [{r['min']}, {r['max']}]",
                         f"width-fit {fit:.4f} / open {open_:.4f} outside the jaw operating range "
                         f"[{r['min']}, {r['max']}]"))
    budget = G.tolerance_budget(use_recess=False)
    out.append(check("tm_tray.tolerance_pitch", budget["pitch_deg"] >= MIN_PITCH_DEG,
                     f"plate may pitch {budget['pitch_deg']:.2f} deg and still enter the corridor",
                     f"plate may pitch only {budget['pitch_deg']:.2f} deg, below {MIN_PITCH_DEG:.2f} deg",
                     budget["pitch_deg"]))
    return out


@register("check", "tm_tray.table")
def table(world) -> list:
    return [check(f"tm_tray.table_confirmed/{e.name}", e.meta["params"]["confirmed"],
                  f"table position measured", "table position is a placeholder (generator defaults); "
                  "every table result is provisional", soft=True)
            for e in _entities(world, "table")]


@register("check", "tm_tray.mount")
def mount(world) -> list:
    out = []
    for e in _entities(world, "pedestal"):
        p = e.meta["params"]
        z = p["base_size"] + p["leg_height"] + p["plate_thickness"]
        slack = z - ORIGIN_Z_THRESHOLD
        out.append(check("tm_tray.mount_height", slack >= 0,
                         f"arm origin at {z:.4f} m, above the {ORIGIN_Z_THRESHOLD:.3f} m threshold "
                         f"({slack * 1000:.0f} mm spare)",
                         f"arm origin at {z:.4f} m, {-slack * 1000:.0f} mm below the {ORIGIN_Z_THRESHOLD:.3f} m "
                         f"threshold; expect the top-back outgoing slots to be unreachable", slack, soft=True))
    return out

"""The TM tray cell built from YAML layouts against the original AI-Robotics-V2 code.

Every expected value is computed by the frozen original ``cell_core`` in
``tests/reference/ai_robotics_v2`` (D22: call across the boundary, do not
transcribe).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from rsf.model.layout import LayoutError, apply_overrides, build_world, load_document, load_world

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ROOT / "cells" / "tm_tray" / "layouts"
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference" / "ai_robotics_v2"))

import cell_core as REF  # noqa: E402
from cell_core import env_adapter as REF_ENV  # noqa: E402
from cell_core import scene as REF_SCENE  # noqa: E402

PRESETS = ["as_built", "recommended", "strict"]
TOL = 1e-12


def ref_name(name: str) -> str:
    """Our solid name for the reference box *name*."""
    head = name.split("/")
    if head[0] == "rack":
        return "/".join(head[1:])
    if head[0] == "mount":
        return "pedestal/" + "/".join(head[1:])
    return name


def builder_width(cfg, rack) -> float:
    """The shelf width the original Isaac builder (and so cell_boxes) used."""
    kw = REF_ENV.generator_kwargs(cfg)
    return (REF.config.min_rack_width(rack.n_jigs, kw["jig_gap"], kw["cylinder_radius"])
            + kw["size_x_margin"])


@pytest.fixture(scope="module", params=PRESETS)
def pair(request):
    """``(cfg, world)``: the original preset and our layout of the same name,
    with each rack's width set to what the original builder produced."""
    cfg = REF.load_preset(request.param)
    doc = load_document(LAYOUTS / f"{request.param}.yaml")
    doc = apply_overrides(doc, {f"entities.{r.name}.params.width": builder_width(cfg, r) for r in cfg.racks})
    return cfg, build_world(doc)


def by_name(solids) -> dict:
    return {s.name: s for s in solids}


def test_structure_matches(pair):
    cfg, world = pair
    mine = by_name(world.solids(objects=False))
    for b in REF_SCENE.rack_boxes(cfg) + REF_SCENE.table_boxes(cfg) + REF_SCENE.mount_boxes(cfg) + [REF_SCENE.floor_box()]:
        s = mine[ref_name(b.name)]
        assert np.allclose(s.T, b.T, atol=TOL, rtol=0), b.name
        assert np.allclose(s.size, b.size, atol=TOL, rtol=0), b.name


def test_jigs_match(pair):
    cfg, world = pair
    for slot in REF.enumerate_slots(cfg):
        ref = sorted((np.round(b.T, 9).tobytes(), np.round(b.size, 9).tobytes()) for b in REF_SCENE.jig_boxes(slot))
        ent = world.entity(f"jig/{slot.key}")
        got = sorted((np.round(s.T, 9).tobytes(), np.round(s.size, 9).tobytes()) for s in ent.solids)
        assert got == ref, slot.key
        # and to 1e-12, pairing each reference box with its nearest
        for b in REF_SCENE.jig_boxes(slot):
            d = min(np.abs(s.T - b.T).max() + np.abs(s.size - b.size).max() for s in ent.solids)
            assert d < TOL, (slot.key, b.name)


def test_trays_fill_the_reference_tray_box(pair):
    """The tray asset is rails + recessed body; their union's box is the original tray box."""
    cfg, world = pair
    for slot in REF.enumerate_slots(cfg):
        ref = REF_SCENE.tray_box(slot)
        obj = world.object_at(slot.key)
        pts = np.vstack([s.corners() for s in obj.world_solids()])
        Rinv = ref.T[:3, :3].T
        local = (pts - ref.T[:3, 3]) @ Rinv.T
        assert np.allclose(local.min(0), -ref.size / 2, atol=TOL), slot.key
        assert np.allclose(local.max(0), ref.size / 2, atol=TOL), slot.key


def test_slots_match(pair):
    cfg, world = pair
    from rsf_plugins.tm_tray.geometry import jig_pose, slots
    ref = REF.enumerate_slots(cfg)
    mine = slots(world)
    assert [s.key for s in mine] == [s.key for s in ref]
    for a, b in zip(mine, ref):
        assert a.role == b.role and a.layer == b.layer and a.column == b.column
        assert np.allclose([a.height, a.lateral, a.label_plane_dist, a.plane_thickness],
                           [b.height, b.lateral, b.label_plane_dist, b.plane_thickness], atol=TOL)
        assert np.allclose(a.approach, b.approach, atol=0)
        assert np.allclose(world.feature(a.key).T, REF_SCENE.jig_pose(b), atol=TOL)
        assert np.allclose(jig_pose(a), REF_SCENE.jig_pose(b), atol=TOL)


def test_robot_base_and_process(pair):
    cfg, world = pair
    assert np.allclose(world.robot.base, cfg.robot.base_pose, atol=TOL)
    assert world.process["standoff_far"] == cfg.standoff_far
    assert world.process["standoff_near"] == cfg.standoff_near
    # The tilt is now 15 deg (user 2026-11-12: kept as insurance; cell_core had 0).
    assert cfg.carry_tilt_deg == 0.0 and world.process["carry_tilt_deg"] == 15.0


def test_grid_ids_match(pair):
    cfg, world = pair
    from rsf_plugins.tm_tray.geometry import grid_ids, slots
    assert grid_ids(slots(world)) == REF.geometry.grid_ids(cfg)


def test_counts(pair):
    cfg, world = pair
    assert len(world.objects) == len(REF.enumerate_slots(cfg)) == 37
    assert world.occupancy() == {s.key: f"tray/{s.key}" for s in REF.enumerate_slots(cfg)}


# ── the measured widths, not the builder's ───────────────────────────────────
@pytest.mark.parametrize("preset", PRESETS)
def test_layout_uses_measured_widths(preset):
    """Without the override the racks are as measured. The original builder made
    the incoming shelf 16 mm wider (one width margin for both racks, D22); only
    the incoming rack's shelves and legs differ, and only across its width."""
    cfg = REF.load_preset(preset)
    world = load_world(LAYOUTS / f"{preset}.yaml")
    mine = by_name(world.solids(objects=False))
    assert builder_width(cfg, cfg.incoming) - cfg.incoming.width == pytest.approx(0.016, abs=1e-12)
    for b in REF_SCENE.rack_boxes(cfg):
        s = mine[ref_name(b.name)]
        if b.name.startswith("rack/outgoing/"):
            assert np.allclose(s.T, b.T, atol=TOL) and np.allclose(s.size, b.size, atol=TOL)
        elif "shelf" in b.name:
            assert s.size[1] == pytest.approx(cfg.incoming.width) and np.allclose(s.T, b.T, atol=TOL)
        else:  # a leg: moved 8 mm inward along the rack width
            d = s.T[:3, 3] - b.T[:3, 3]
            assert abs(np.linalg.norm(d) - 0.008) < 1e-12 and abs(d[2]) < 1e-12


# ── layout mechanics ─────────────────────────────────────────────────────────
def test_presets_differ_only_in_leg_height():
    docs = {p: load_document(LAYOUTS / f"{p}.yaml") for p in PRESETS}
    base = docs["as_built"]
    for p in ("recommended", "strict"):
        d = dict(docs[p])
        assert d["entities"]["pedestal"]["params"]["leg_height"] != base["entities"]["pedestal"]["params"]["leg_height"]
        d = apply_overrides(d, {"entities.pedestal.params.leg_height":
                                base["entities"]["pedestal"]["params"]["leg_height"],
                                "name": base["name"], "description": base["description"]})
        assert d == base


def test_override_of_a_missing_path_is_an_error():
    doc = load_document(LAYOUTS / "strict.yaml")
    with pytest.raises(LayoutError):
        apply_overrides(doc, {"entities.pedestal.params.leg_heigth": 0.6})


def test_unknown_generator_parameter_is_an_error():
    doc = load_document(LAYOUTS / "strict.yaml")
    doc["entities"]["incoming"]["params"]["n_jig"] = 3
    with pytest.raises(ValueError, match="n_jig"):
        build_world(doc)


def test_occupancy_selects_trays():
    world = load_world(LAYOUTS / "strict.yaml", occupancy={"tray": ["incoming/L0/C0", "table/L0/C0"]})
    assert sorted(world.occupancy()) == ["incoming/L0/C0", "table/L0/C0"]
    with pytest.raises(LayoutError):
        load_world(LAYOUTS / "strict.yaml", occupancy={"tray": ["nowhere/L0/C0"]})


def test_tray_offset_is_the_jig_column_top():
    """The layout seats the tray at 0.035; the jig asset's column tops must agree."""
    from rsf.model.urdf import UrdfModel
    from rsf.model.world import solids_from_urdf
    doc = load_document(LAYOUTS / "as_built.yaml")
    jig = solids_from_urdf(UrdfModel.from_file(doc["placements"]["jig"]["urdf"]))
    cols = [s for s in jig if abs(s.T[2, 3] - 0.02) < 1e-9 and abs(s.size[2] - 0.03) < 1e-9]
    assert len(cols) == 4
    tops = {round(s.T[2, 3] + s.size[2] / 2, 12) for s in cols}
    assert tops == {doc["objects"]["tray"]["offset"]["xyz"][2]}

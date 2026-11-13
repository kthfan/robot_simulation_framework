"""Workflow expansion, the audit, selectors and the cuRobo scene, on the TM tray cell.

Expected orders and verdicts come from the frozen original ``cell_core``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from rsf.compile.curobo import world_scene
from rsf.core import se3
from rsf.model import audit as AU
from rsf.model.layout import apply_overrides, build_world, load_document, load_world
from rsf.model.select import SelectorError, select_features
from rsf.model.workflow import Workflow, WorkflowError, expand

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "cells" / "tm_tray"
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference" / "ai_robotics_v2"))
import cell_core as REF  # noqa: E402
from cell_core import audit as REF_AUDIT  # noqa: E402
from cell_core import scene as REF_SCENE  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return load_world(CELL / "layouts" / "strict.yaml")


@pytest.fixture(scope="module")
def wf():
    return Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")


# ── workflow ─────────────────────────────────────────────────────────────────
def test_inspect_and_sort_follows_the_original_scheduler(world, wf):
    cfg = REF.load_preset("strict")
    ref = REF.SequentialScheduler(REF.enumerate_slots(cfg))
    sources = [s.key for s in ref.order()]
    dests = [s.key for s in ref.destinations()]
    ex = expand(wf, world)
    assert ex.cycles == len(sources) == 12
    assert len(ex.tasks) == 3 * 12
    for c in range(ex.cycles):
        t = [x for x in ex.tasks if x.cycle == c]
        assert [x.skill for x in t] == ["tm_tray.transfer", "wait", "tm_tray.transfer"]
        assert t[0].args == {"object": "tray", "from": sources[c], "to": "table/L0/C0"}
        assert t[2].args == {"object": "tray", "from": "table/L0/C0", "to": dests[c]}
    assert ex.end.name == "request_refill"
    assert sorted(ex.occupancy) == sorted(dests[:12])


def test_full_outgoing_rack_requests_removal(world, wf):
    sets = wf.resolve_sets(world)
    occ = {f.key: f"tray/{f.key}" for f in sets["incoming"] + sets["outgoing"][:-2]}
    ex = expand(wf, world, occupancy=occ)
    assert ex.cycles == 2
    assert ex.end.name == "request_removal"


def test_tray_identity_is_kept(world, wf):
    ex = expand(wf, world)
    first = ex.tasks[0].args["from"]
    assert ex.occupancy[ex.tasks[2].args["to"]] == f"tray/{first}"


def test_unknown_binding_is_an_error(world):
    doc = {"name": "x", "sets": {"a": {"kind": "slot"}},
           "scheduler": {"type": "rsf.ordered", "params": {"bind": {"s": {"set": "a"}}}},
           "cycle": [{"skill": "noop", "args": {"at": "$nope"}}]}
    with pytest.raises(WorkflowError):
        expand(Workflow.from_dict(doc), world)


def test_set_used_as_a_single_target_must_hold_one_feature(world):
    doc = {"name": "x", "sets": {"a": {"kind": "slot", "role": "incoming"}},
           "scheduler": {"type": "rsf.ordered", "params": {"bind": {"s": {"set": "a"}}}},
           "cycle": [{"skill": "noop", "args": {"at": "$a"}}], "max_cycles": 1}
    with pytest.raises(WorkflowError, match="exactly one"):
        expand(Workflow.from_dict(doc), world)


# ── selectors ────────────────────────────────────────────────────────────────
def test_selectors(world):
    slots = world.features({"kind": "slot"})
    assert len(select_features(slots, "incoming/L0/*")) == 4
    assert len(select_features(slots, {"role": "outgoing", "layer": [0, 3]})) == 12
    assert len(select_features(slots, {"z": {"min": 1.0}, "role": "incoming"})) == 8
    assert [f.key for f in select_features(slots, ["table/*", "incoming/L0/C0", "table/*"])] == \
        ["table/L0/C0", "incoming/L0/C0"]
    with pytest.raises(SelectorError):
        select_features(slots, {"z": {"low": 1}})


# ── audit ────────────────────────────────────────────────────────────────────
STATUS_MAP = {  # original check name -> ours (for the checks that were ported)
    "rack_width/incoming": "tm_tray.rack_width/incoming",
    "rack_width/outgoing": "tm_tray.rack_width/outgoing",
    "layer_gap/incoming": "tm_tray.layer_gap/incoming",
    "layer_gap/outgoing": "tm_tray.layer_gap/outgoing",
    "standoff_plate_free": "tm_tray.standoff_plate_free",
    "grid_in_view/near": "tm_tray.grid_in_view/near",
    "grid_in_view/far": "tm_tray.grid_in_view/far",
    "grid_pixels/near": "tm_tray.grid_pixels/near",
    "grid_pixels/far": "tm_tray.grid_pixels/far",
    "tolerance_pitch": "tm_tray.tolerance_pitch",
    "table_confirmed": "tm_tray.table_confirmed/table",
    "mount_height": "tm_tray.mount_height",
}


@pytest.mark.parametrize("preset", ["as_built", "recommended", "strict"])
def test_audit_agrees_with_the_original(preset):
    ref = {c.name: c for c in REF_AUDIT.audit(REF.load_preset(preset))}
    mine = {c.name: c for c in AU.audit(load_world(CELL / "layouts" / f"{preset}.yaml"))}
    for r, m in STATUS_MAP.items():
        assert mine[m].status == ref[r].status, (r, mine[m].detail, ref[r].detail)
        if ref[r].value is not None:
            assert mine[m].value == pytest.approx(ref[r].value, abs=1e-12), r
    assert not AU.skipped(mine.values())
    assert mine["rsf.unique_solid_names"].status == AU.PASS


def test_audit_catches_a_bad_layout():
    doc = load_document(CELL / "layouts" / "strict.yaml")
    doc = apply_overrides(doc, {"entities.outgoing.params.width": 1.0,
                                "entities.incoming.params.layer_heights": [0.82, 0.90],
                                "process.standoff_near": 0.2,
                                "process.carry_tilt_deg": 45.0,
                                "process.jaw_operating_range": {"min": 0.045, "max": 0.06}})
    checks = AU.audit(build_world(doc))
    failed = {c.name for c in AU.failures(checks)}
    assert {"tm_tray.rack_width/outgoing", "tm_tray.layer_gap/incoming", "tm_tray.standoff_plate_free",
            "tm_tray.carry_tilt", "tm_tray.jaw_range"} <= failed
    assert AU.verdict(checks) == AU.FAIL


def test_audit_never_raises():
    from rsf.model.registry import register

    @register("check", "test.boom", replace=True)
    def boom(world):
        raise RuntimeError("kaput")

    checks = AU.audit(load_world(CELL / "layouts" / "strict.yaml"), only=["test.boom"])
    assert [c.status for c in checks] == [AU.SKIP] and "kaput" in checks[0].detail


# ── cuRobo scene ─────────────────────────────────────────────────────────────
def test_curobo_scene_matches_the_original():
    """Our box obstacles, in the base frame, equal cell_core.scene.to_curobo_scene
    (racks at the original builder width)."""
    from test_tm_tray_layout import builder_width, ref_name
    cfg = REF.load_preset("strict")
    doc = apply_overrides(load_document(CELL / "layouts" / "strict.yaml"),
                          {f"entities.{r.name}.params.width": builder_width(cfg, r) for r in cfg.racks})
    world = build_world(doc)
    mine = world_scene(world, objects=False)["cuboid"]
    boxes = REF_SCENE.rack_boxes(cfg) + REF_SCENE.table_boxes(cfg) + REF_SCENE.mount_boxes(cfg) + [REF_SCENE.floor_box()]
    ref = REF_SCENE.to_curobo_scene(boxes, cfg.robot.base_pose)["cuboid"]
    for name, r in ref.items():
        m = mine[ref_name(name)]
        assert np.allclose(m["dims"], r["dims"], atol=1e-12)
        assert np.allclose(m["pose"][:3], r["pose"][:3], atol=1e-12)
        assert se3.pose_error(se3.make_T(se3.quat_to_matrix(m["pose"][3:])),
                              se3.make_T(se3.quat_to_matrix(r["pose"][3:])))[1] < 1e-9


def test_curobo_scene_counts(world):
    sc = world_scene(world)
    assert set(sc) == {"cuboid"}               # every solid in this cell is a box
    assert len(sc["cuboid"]) == len(world.solids())
    empty = world_scene(world, exclude_objects=["tray/incoming/L0/C0"])
    assert len(empty["cuboid"]) == len(sc["cuboid"]) - 3


def test_bindings_never_share_a_feature(world):
    """Two bindings drawing on the same set (e.g. a reshuffle inside one rack) get
    different features in the same cycle."""
    doc = {"name": "x", "sets": {"r": {"kind": "slot", "role": "incoming"}},
           "scheduler": {"type": "rsf.ordered", "params": {"bind": {
               "a": {"set": "r", "order": ["key"], "state": "any"},
               "b": {"set": "r", "order": ["key"], "state": "any"}}}},
           "cycle": [{"skill": "noop", "args": {"x": "$a", "y": "$b"}}], "max_cycles": 1}
    t = expand(Workflow.from_dict(doc), world).tasks[0]
    assert t.args["x"] != t.args["y"]

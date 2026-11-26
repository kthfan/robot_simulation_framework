"""The example cell (cells/example_pick_place, docs/new_cell.md) end to end, as a new
cell's author runs it: audit -> expand -> teach -> plan (CPU) -> simulate (MuJoCo) -> judge.

It is the framework's own smoke test that nothing in it is TM-tray specific, and the
template a new cell's test can copy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rsf.model import audit as AU
from rsf.model.layout import load_world
from rsf.model.workflow import Workflow, expand

pytest.importorskip("eaik")
pytest.importorskip("coal")

CELL = Path(__file__).resolve().parents[1] / "cells" / "example_pick_place"
LAYOUT = CELL / "layouts" / "main.yaml"
WORKFLOW = CELL / "workflows" / "move_blocks.yaml"


@pytest.fixture(scope="module")
def cell():
    world = load_world(LAYOUT)
    wf = Workflow.from_file(WORKFLOW)
    return world, wf


def test_audit_and_expand(cell):
    world, wf = cell
    checks = AU.audit(world)
    assert AU.verdict(checks) == AU.PASS, [c for c in checks if c.status != AU.PASS]
    assert {c.name for c in checks} >= {"example.stands.spacing", "example.stands.roles"}
    ex = expand(wf, world)
    assert [(t.args["from"], t.args["to"]) for t in ex.tasks] == [("stand_a/top", "stand_c/top"),
                                                                ("stand_b/top", "stand_d/top")]
    assert ex.end.name == "sources_empty"


def test_generator_rejects_unknown_parameters():
    from rsf.model import registry
    with pytest.raises(ValueError, match="unknown parameter"):
        registry.get("generator", "example.stand")("s", {"xy": [0, 0], "hieght": 0.2}, None)


def test_one_cycle_teach_plan_simulate_judge(cell):
    """The first cycle (one transfer) through the whole pipeline; every stage's verdict passes."""
    pytest.importorskip("mujoco")
    from rsf.planning import plan as PL
    from rsf.planning.teach import make_engine, teach_workflow
    from rsf.sim import judge as J
    from rsf.sim.mujoco_cell import MujocoCell
    world, wf = cell
    wf = Workflow.from_dict({**_doc(), "max_cycles": 1})
    eng = make_engine(world)
    taught = teach_workflow(world, wf, eng)
    seqs = taught["sequences"]
    assert set(seqs) == {"pick:stand_a/top", "place:stand_c/top"}
    assert all(r.ok for r in seqs.values()), {n: r.problems for n, r in seqs.items()}
    grasp = next(p for p in seqs["pick:stand_a/top"].points if p.name == "grasp")
    assert grasp.clearance >= 0.0 and grasp.arrive == "line"

    teach = {n: [vars(p) for p in r.points] for n, r in seqs.items()}
    segs = PL.segments_for_workflow(world, wf, teach)
    assert [s.kind for s in segs].count("free") == 3                 # home -> pick, pick -> place, place -> home
    mo = PL._motion(world.process)
    for s in segs:
        if s.kind == "free":
            assert PL.plan_free_gated(eng.arm, eng.checker, s, mo)["success"], s.name
    traj, problems = PL.assemble(world, eng.arm, eng.checker, segs)
    assert problems == []
    carry = next(s for s in segs if s.kind == "free" and s.carried is not None)
    assert carry.caps == "carry"

    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    doc = traj.to_dict()
    rep = MujocoCell(world, occ).run(doc)
    res = J.judge(world, wf, rep, occ, traj=doc)
    assert res["ok"], res
    (tr,) = res["transfers"]
    assert (tr["src"], tr["dst"]) == ("stand_a/top", "stand_c/top")
    assert tr["seat_mm"] < 1.0 and tr["slip_mm"] < 2.0 and tr["track_mm"] < 1.0
    # the block really travelled: it ends on the target stand, 0.45 m from where it started
    block = rep.objects[tr["object"]]
    assert np.linalg.norm(block[-1][:3, 3] - block[0][:3, 3]) == pytest.approx(0.45, abs=0.002)


def _doc():
    import yaml
    return yaml.safe_load(WORKFLOW.read_text())


def test_grip_carry_rule_has_no_outward_pull(cell):
    """``carry_support: grip``: only the tilt cone counts (a clamped object may hang below)."""
    from rsf.kinematics import load_arm
    from rsf.planning import plan as PL
    world, _ = cell
    arm = load_arm(world)
    rule = PL.carry_rule(PL._motion(world.process))
    assert rule == {"max_tilt_deg": 30.0, "reference": "start", "support": "grip"}
    q0 = np.radians([130.45, -1.55, -100.76, 12.34, 269.97, 40.45])
    held = ("block", "tcp", world.object_at("stand_a/top").solids, np.eye(4))   # centred on the tcp
    s = PL.Segment("free", "free", {}, q_start=q0, q_goal=q0, carried=held)
    g_nom, c_dir, cos_max = PL.carry_frame(arm, s, **rule)
    assert c_dir is None and cos_max == pytest.approx(math.cos(math.radians(30.0)))
    assert PL.carry_violation(arm, q0, (g_nom, c_dir, cos_max)) is None
    axis, angle = PL.carry_bound((g_nom, c_dir, cos_max))
    assert np.allclose(axis, -g_nom) and angle == pytest.approx(math.radians(30.0))
    tilted = q0 + np.radians([0, 0, 0, 0, 40, 0])
    assert "tilted" in PL.carry_violation(arm, tilted, (g_nom, c_dir, cos_max))
    with pytest.raises(ValueError, match="carry_support"):
        PL.carry_rule({"carry_support": "hover"})


def test_judge_settings(cell):
    from dataclasses import replace

    from rsf.sim import judge as J
    world, _ = cell
    assert J.settings(world) == {"carry_from": "grasp:act", "carry_to": "place_down", "track_caps": ("near",)}
    with pytest.raises(ValueError, match="unknown"):
        J.settings(replace(world, process={"judge": {"carry_form": "x"}}))

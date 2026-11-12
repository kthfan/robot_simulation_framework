"""The M3 collision check on the CPU (port of AI-Robotics-V2 check_cell).

Fast tests run four representative slots; the full 37-slot runs are marked slow
(``pytest -m slow``, about a minute each).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("eaik")

from rsf.model.layout import load_world  # noqa: E402
from rsf.planning.chain import ContinuityRule, best_chain, chain_joints, turns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "cells" / "tm_tray"
NO_AIR = str((CELL / "assets" / "tm5s_egu50_no_air_cylinder.urdf").resolve())
SAMPLE = ["incoming/L0/C0", "outgoing/L3/C0", "outgoing/L3/C5", "table/L0/C0"]
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference" / "ai_robotics_v2"))


@pytest.fixture(scope="module")
def no_air():
    return load_world(CELL / "layouts" / "strict.yaml", {"robot.urdf": NO_AIR})


@pytest.fixture(scope="module")
def with_air():
    return load_world(CELL / "layouts" / "strict.yaml")


def test_carried_tray_is_the_original_carried_box(no_air):
    import cell_core as REF
    from cell_core import scene as RS, sequence as RQ
    from rsf_plugins.tm_tray.collision_check import carried_tray_pose
    cfg = REF.load_preset("strict")
    for s in REF.enumerate_slots(cfg)[::7]:
        T = RQ.grasp_sequence(cfg, s)["lift"]
        ref = RS.carried_tray_box(T)
        pts = np.vstack([x.moved(T @ carried_tray_pose(RQ.insert_tip_depth())).corners()
                         for x in no_air.object_at(s.key).solids])
        loc = (pts - ref.T[:3, 3]) @ ref.T[:3, :3]
        assert np.allclose(loc.min(0), -ref.size / 2, atol=1e-12) and np.allclose(loc.max(0), ref.size / 2, atol=1e-12)


def test_d29_gripper_passes(no_air):
    """D29: with the D29 gripper the strict cell passes (sample of the 37)."""
    from rsf_plugins.tm_tray.collision_check import check_cell
    r = check_cell(no_air, select=SAMPLE)
    assert (r["passed"], r["total"]) == (4, 4), {s["slot"]: s["missing"] for s in r["slots"]}
    for s in r["slots"]:
        qs = np.array(s["chain_q"])
        assert qs.shape == (3, 6)
        assert np.all(np.abs(np.diff(qs, axis=0)) <= 1.0 + 1e-9)


def test_without_the_push_plate_stop_the_plate_hits_the_tray(with_air):
    """At the D28 insert depth the installed push plate reaches 9.5 mm into the
    target tray (the push-plate stop switched off: ``process.push_plate: {}``)."""
    from rsf_plugins.tm_tray.collision_check import check_cell
    world = load_world(CELL / "layouts" / "strict.yaml", {"process.push_plate": {}})
    r = check_cell(world, select=["incoming/L0/C0"])
    s = r["slots"][0]
    assert not s["ok"] and s["missing"] == ["insert"]
    tray_hits = [v for b in s["phases"]["insert"] for v in b["violations"]
                 if v[1].startswith("tray/incoming/L0/C0/")]
    assert tray_hits and {v[0] for v in tray_hits} == {"gripper_body"}
    assert min(v[2] for v in tray_hits) == pytest.approx(-0.0095, abs=2e-4)


def test_push_plate_stop_keeps_2mm_and_passes(with_air):
    """With process.push_plate.margin = 2 mm the sample slots pass, and the exact
    meshes put the push plate 2 mm from the target tray at insert."""
    from rsf.kinematics import load_arm
    from rsf.model.urdf import UrdfModel
    from rsf.model.world import solids_from_urdf
    from rsf_plugins.tm_tray.collision_check import check_cell
    from rsf_plugins.tm_tray.geometry import slot_of
    from rsf_plugins.tm_tray.sequence import grasp_sequence
    from rsf_plugins.tm_tray.tool import tool_geometry
    assert with_air.process["push_plate"]["margin"] == 0.002
    r = check_cell(with_air, select=SAMPLE)
    assert (r["passed"], r["total"]) == (4, 4), {s["slot"]: s["missing"] for s in r["slots"]}
    arm = load_arm(with_air)
    key = "outgoing/L1/C2"
    T = grasp_sequence(with_air.process, slot_of(with_air.feature(key)), tool=tool_geometry(with_air))["insert"]
    q = arm.ik(T, expand_turns=False)[0]
    model = UrdfModel.from_file(with_air.robot.urdf)
    plate = [s for s in solids_from_urdf(model) if s.name.endswith("air_cylinder_plate")]
    fk = model.fk(dict(zip(arm.joint_names, q)))
    pose = with_air.robot.base @ fk["gripper_body"]
    plate_world = [s.moved(pose @ np.linalg.inv(model.fk()["gripper_body"])) for s in plate]
    tray = with_air.object_at(key).world_solids()
    from rsf.collision.coal_backend import make_geometry, signed_distance
    d = min(signed_distance(make_geometry(p.shape)[0], p.T, make_geometry(t.shape)[0], t.T)
            for p in plate_world for t in tray)
    assert d == pytest.approx(0.002, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("robot,expect", [(NO_AIR, 37), (None, 37)], ids=["no_air_cylinder", "air_cylinder"])
def test_full_cell(robot, expect):
    world = load_world(CELL / "layouts" / "strict.yaml", {"robot.urdf": robot} if robot else None)
    from rsf_plugins.tm_tray.collision_check import check_cell  # loaded by the layout
    assert check_cell(world)["passed"] == expect


# ── chain continuity (generic) ───────────────────────────────────────────────
def test_turns_stay_in_limits():
    lo, hi = np.full(2, -2 * np.pi), np.full(2, 2 * np.pi)
    forms = turns(np.array([3.0, -1.0]), lo, hi)
    assert len(forms) == 4 and all(np.all((f >= lo) & (f <= hi)) for f in forms)


def test_chain_takes_a_whole_turn_when_it_must():
    """AI-Robotics-V2 D28: -5.20 and 1.08 rad are the same joint 1."""
    lo, hi = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
    a = np.array([-5.20, 0.1, 1.0, 0.0, 1.0, 0.0])
    b = np.array([1.08, 0.1, 1.0, 0.0, 1.0, 0.0])
    qs = chain_joints([a, b], lo, hi)
    assert qs is not None and abs(qs[0][0] - qs[1][0]) < 1.0


def test_chain_refuses_an_elbow_flip():
    lo, hi = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
    a = np.array([0.0, 0.1, 0.2, 0.0, 1.0, 0.0])
    b = np.array([0.0, 0.1, -0.2, 0.0, 1.0, 0.0])
    assert chain_joints([a, b], lo, hi, ContinuityRule()) is None


def test_best_chain_prefers_the_larger_worst_clearance():
    lo, hi = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
    q = np.array([0.0, 0.1, 1.0, 0.0, 1.0, 0.0])
    cands = [[{"q": q, "ok": True, "score": 0.01}, {"q": q + 0.1, "ok": True, "score": 0.02}],
             [{"q": q + 0.2, "ok": True, "score": 0.03}, {"q": q, "ok": False, "score": 0.5}]]
    idx, _ = best_chain(cands, lo, hi)
    assert idx == (1, 0)

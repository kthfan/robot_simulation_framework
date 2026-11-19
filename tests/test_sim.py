"""P5: MuJoCo replay and the D30 judge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rsf.core import se3  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402
from rsf.model.workflow import Workflow  # noqa: E402
from rsf.sim import judge as J  # noqa: E402
from rsf.sim.mujoco_cell import ExecReport, MujocoCell  # noqa: E402

CELL = Path(__file__).resolve().parents[1] / "cells" / "tm_tray"
SUMMARY = CELL / "plans" / "strict_inspect_and_sort_gpu.summary.json"


@pytest.fixture(scope="module")
def setup():
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    return world, wf, occ


def test_transfers_from_the_plan(setup):
    world, wf, occ = setup
    segs = json.loads(SUMMARY.read_text())["segments"]
    trs = J.transfers(world, wf, segs, occ)
    assert len(trs) == 24
    a, b = trs[0], trs[1]
    assert (a.src, a.dst, b.src, b.dst) == ("incoming/L2/C0", "table/L0/C0", "table/L0/C0", "outgoing/L0/C0")
    assert a.obj == b.obj == "tray/incoming/L2/C0"                  # the same tray, tracked
    assert a.t0 < a.carry[0] < a.carry[1] < a.t1 <= b.t0
    # placed 2 mm shallower than the seat (process.push_plate.place_offset), toward the robot (+y)
    d = np.linalg.inv(world.object_at("table/L0/C0").T) @ a.seat_T
    assert np.allclose(d[:3, 3], [0.0, 0.002, 0.0], atol=1e-9)


def _report(world, wf, occ, tray_offset=np.zeros(3), contact=None):
    """A synthetic run of the first transfer: the tray rides 0.25 m in front of the flange."""
    segs = json.loads(SUMMARY.read_text())["segments"]
    tr = J.transfers(world, wf, segs, occ)[0]
    t = np.arange(0.0, tr.t1 + 0.02, 0.02)
    fl = np.array([se3.make_T(None, (0.3 + 0.001 * k, 0.0, 1.0)) for k in range(len(t))])
    obj = []
    for k, tt in enumerate(t):
        if tt < tr.carry[0]:
            obj.append(world.object_at(tr.src).T)
        elif tt <= tr.carry[1]:
            obj.append(fl[k] @ se3.make_T(None, (0.0, 0.0, 0.25)))
        else:
            obj.append(tr.seat_T @ se3.make_T(None, tray_offset))
    return ExecReport(t, fl, fl.copy(), np.zeros((len(t), 8)), {tr.obj: np.array(obj)},
                      [] if contact is None else [contact], segs[: [s["name"] for s in segs].index("hold:wait") + 1])


def test_judge_pass_and_each_failure(setup):
    world, wf, occ = setup
    ok = J.judge(world, wf, _report(world, wf, occ), occ)
    assert ok["ok"] and len(ok["transfers"]) == 1, ok
    off = J.judge(world, wf, _report(world, wf, occ, tray_offset=(0.0, 0.003, 0.0)), occ)["transfers"][0]
    assert not off["checks"]["placement"] and off["seat_mm"] == pytest.approx(3.0, abs=1e-6)
    hit = J.judge(world, wf, _report(world, wf, occ, contact={"t": 5.0, "link": "gripper_body",
                                                                 "other": "rack/incoming/leg0", "depth": 0.001}), occ)
    assert not hit["transfers"][0]["checks"]["contacts"]
    assert hit["transfers"][0]["contacts"][0]["depth_mm"] == pytest.approx(1.0)
    own = J.judge(world, wf, _report(world, wf, occ, contact={"t": 5.0, "link": "gripper_body",
                                                                 "other": "tray/incoming/L2/C0", "depth": 0.001}), occ)
    assert own["ok"]                                             # touching the tray it transfers is fine


def test_cell_holds_still(setup):
    """The arm held at home for 2 s: trays stay seated, no robot contacts, tracking < 0.1 mm."""
    world, wf, occ = setup
    cell = MujocoCell(world, occ)
    assert sorted(cell.object_names) == sorted(world.object_at(k).name for k in occ)
    traj = {"joint_names": [f"joint_{i}" for i in range(1, 7)] + ["left_gripper_joint", "right_gripper_joint"],
            "t": [0.0, 2.0], "q": [[0.0] * 6 + [0.0577] * 2] * 2, "qd": [[0.0] * 8] * 2}
    rep = cell.run(traj)
    assert not rep.contacts
    for n, P in rep.objects.items():
        o = next(o for o in world.objects if o.name == n)
        assert max(J.pose_error(P[-1], o.T)) < 0.2, n           # mm / deg
    assert max(J.pose_error(a, b)[0] for a, b in zip(rep.flange, rep.flange_planned)) < 0.1


PLAN = CELL / "plans" / "strict_inspect_and_sort_gpu.json"


@pytest.mark.slow
@pytest.mark.skipif(not PLAN.exists(), reason="run tools/plan_workflow.py assemble first")
def test_first_cycle_passes(setup):
    """The planned strict run, first cycle (2 transfers), in MuJoCo: every D30 criterion."""
    world, wf, occ = setup
    traj = json.loads(PLAN.read_text())
    trs = J.transfers(world, wf, traj["segments"], occ)
    rep = MujocoCell(world, occ).run(traj, t_end=trs[1].t1 + 0.5)
    res = J.judge(world, wf, rep, occ)
    assert res["ok"] and len(res["transfers"]) == 2, res

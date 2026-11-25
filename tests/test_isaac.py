"""P5: the Isaac Sim cell (PhysX). Needs Isaac Sim and a GPU: skipped elsewhere,
run on Kaggle (``tools/isaac/kaggle_check.sh``). The same checks as the MuJoCo cell."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("isaacsim")

from rsf.model.layout import load_world  # noqa: E402
from rsf.model.workflow import Workflow  # noqa: E402
from rsf.sim import judge as J  # noqa: E402
from rsf.sim.isaac_cell import IsaacCell  # noqa: E402

CELL = Path(__file__).resolve().parents[1] / "cells" / "tm_tray"
CYCLE1 = CELL / "plans" / "strict_cycle1.json.gz"


@pytest.fixture(scope="module")
def setup():
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    return world, wf, occ, IsaacCell(world, occ)


def test_cell_holds_still(setup):
    """The arm held at home for 2 s: trays stay seated, no robot contacts, tracking < 0.1 mm."""
    world, wf, occ, cell = setup
    traj = {"joint_names": [f"joint_{i}" for i in range(1, 7)] + ["left_gripper_joint", "right_gripper_joint"],
            "t": [0.0, 2.0], "q": [[0.0] * 6 + [0.0577] * 2] * 2, "qd": [[0.0] * 8] * 2}
    rep = cell.run(traj)
    assert not rep.contacts
    for n, P in rep.objects.items():
        o = next(o for o in world.objects if o.name == n)
        assert max(J.pose_error(P[-1], o.T)) < 0.2, n
    assert max(J.pose_error(a, b)[0] for a, b in zip(rep.flange, rep.flange_planned)) < 0.1


@pytest.mark.slow
def test_first_cycle_passes(setup):
    world, wf, occ, cell = setup
    traj = json.loads(gzip.decompress(CYCLE1.read_bytes()))
    trs = J.transfers(world, wf, traj["segments"], occ)
    rep = cell.run(traj, t_end=trs[1].t1 + 0.5)
    res = J.judge(world, wf, rep, occ, traj=traj)
    assert res["ok"] and len(res["transfers"]) == 2, res


def test_live_view_renders_and_leaves_the_run(setup):
    """RTX views (rsf.live): pictures of the moving arm, and the run as without them."""
    import numpy as np

    from rsf.live.capture import LiveCapture
    from rsf.live.hub import Hub
    from rsf.sim.common import overview
    world, wf, occ, cell = setup
    names = [f"joint_{i}" for i in range(1, 7)] + ["left_gripper_joint", "right_gripper_joint"]
    q0 = [0.0] * 6 + [0.0577] * 2
    q1 = [0.6, -0.3] + [0.0] * 4 + [0.0577] * 2
    traj = {"joint_names": names, "t": [0.0, 2.0], "q": [q0, q1], "qd": [[0.0] * 8] * 2}
    a = cell.run(traj)
    hub = Hub()
    cap = LiveCapture(hub, cell, [overview(world, width=480, height=270)], fps=4.0)
    cap.warmup()
    first = [None]

    def hook(t, *x):
        cap(t, *x)
        if first[0] is None and t >= 0.0:
            first[0] = cell.render("overview")
    b = cell.run(traj, on_sample=hook)
    last = cell.render("overview")
    cap.finish()
    c = cell.run(traj)
    tol = max(1e-9, 2.0 * float(np.abs(a.q - c.q).max()))          # run-to-run spread without rendering
    assert float(np.abs(a.q - b.q).max()) <= tol
    assert first[0] is not None and first[0].shape == (270, 480, 3) and first[0].std() > 5
    assert np.abs(last.astype(int) - first[0].astype(int)).mean() > 0.5      # the arm moved in the picture
    assert hub.streams["overview"].seq >= 2

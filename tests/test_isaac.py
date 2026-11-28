"""P5: the Isaac Sim cell (PhysX). Needs Isaac Sim and a GPU: skipped elsewhere,
run on Kaggle (``tools/isaac/kaggle_check.sh``). The same checks as the MuJoCo cell."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("isaacsim", reason="needs Isaac Sim's Python: the Isaac venv, or python.sh of the standalone package")

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
    assert cell.render_lag() < 0.005               # what the renderer draws is where PhysX has the arm
    last = cell.render("overview")
    cap.finish()
    c = cell.run(traj)
    # app updates round PhysX's float32 state (an ULP, Isaac Sim 5.0): the flange within 0.05 mm
    assert float(np.abs(a.flange[:, :3, 3] - b.flange[:, :3, 3]).max()) <= max(5e-5, 2.0 * float(
        np.abs(a.flange[:, :3, 3] - c.flange[:, :3, 3]).max()))
    assert first[0] is not None and first[0].shape == (270, 480, 3) and first[0].std() > 5
    d = np.abs(last.astype(int) - first[0].astype(int)).max(axis=2)
    assert (d > 10).mean() > 0.002                 # the arm moved in the picture
    assert hub.streams["overview"].seq >= 2
    cell.move_view("overview", (0.0, -3.0, 1.6), (0.0, 0.3, 0.9), 60.0)      # a viewer moves the camera
    for _ in range(3):
        moved = cell.render("overview")
        cell.move_view("overview", (0.0, -3.0, 1.6), (0.0, 0.3, 0.9), 60.0)
    assert (np.abs(moved.astype(int) - last.astype(int)).max(axis=2) > 10).mean() > 0.1


def test_read_camera_renders_the_labels(tmp_path):
    """The locate tour's reads with RTX (rsf.sim.reads): every read of the first two
    slots gives a sharp image of its label (black tags on white). With the detector
    installed, the board is found where the cell has it."""
    import numpy as np

    from rsf.sim import reads as RD
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    traj = json.loads(gzip.decompress(CYCLE1.read_bytes()))
    setup = RD.setup_for(world)
    cell = IsaacCell(world, occ, reads=setup)
    rd = RD.Reader(setup, traj["segments"], cell, images=tmp_path)
    t_end = rd.todo[3]["t"] + 0.1                        # far and near reads of two slots
    cell.run(traj, t_end=t_end, on_sample=rd)
    assert len(rd.rows) == 4
    from PIL import Image
    for r in rd.rows:
        img = np.asarray(Image.open(tmp_path / r["image"]))
        assert img.shape == (setup.cameras[r["camera"]].height, setup.cameras[r["camera"]].width)
        assert (img < 60).mean() > 0.002 and (img > 190).mean() > 0.01, r["image"]
        if r["found"] is not None:
            assert r["found"] and r["pos_err_mm"] < 0.5 and r["rot_err_deg"] < 0.5, r

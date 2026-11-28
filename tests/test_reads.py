"""P6 reads during a run (rsf.sim.reads): the plan's read holds, the camera image and
board the simulated cell gives there, and the racks located from those reads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("eaik")
pytest.importorskip("coal")
pytest.importorskip("aprilgrid_detector")

from rsf.model.layout import load_world  # noqa: E402
from rsf.sim import reads as RD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "cells" / "tm_tray"
STRICT = CELL / "layouts" / "strict.yaml"
MOVED = CELL / "layouts" / "perturbed" / "strict_racks_moved.yaml"
SLOTS = ("incoming/L2/C0", "incoming/L2/C3", "incoming/L0/C0", "table/L0/C0")


@pytest.fixture(scope="module")
def taught():
    """The locate sequences of a few slots, taught in the nominal (strict) layout."""
    from rsf.model import registry
    from rsf.planning.teach import make_engine
    world = load_world(STRICT)
    eng = make_engine(world)
    out = {}
    for seq in registry.get("skill", "tm_tray.locate")(world, {"at": list(SLOTS)}):
        r = eng.teach(seq)
        assert r.ok, r.problems
        out[seq.meta["slot"]] = {p.name: p for p in r.points}
    return world, out


def _segments(keys):
    """Hold segments as a plan writes them: one second per read, plus a move that is no read."""
    segs, t = [{"name": "free001:locate:x/sight_far", "kind": "free", "t0": 0.0, "t1": 1.0}], 1.0
    for k in keys:
        for read in ("sight_far", "sight_near"):
            segs.append({"name": f"locate:{k}/{read}:act", "kind": "hold", "t0": t, "t1": t + 1.0})
            t += 1.0
    return segs


def _run(world_sim, taught_pts, keys):
    """Every read of *keys*, the simulated arm held at the taught read pose."""
    from rsf.compile.mjcf import ROBOT_PREFIX
    from rsf.sim.mujoco_cell import MujocoCell
    setup = RD.setup_for(world_sim)
    cell = MujocoCell(world_sim, reads=setup)
    rd = RD.Reader(setup, _segments(keys), cell)
    sched = list(rd.todo)
    for r in sched:
        p = taught_pts[r["key"]][r["read"]]
        for i, v in enumerate(p.q):
            cell.d.qpos[cell.m.joint(ROBOT_PREFIX + f"joint_{i + 1}").qposadr[0]] = v
        rd(r["t"])
    return rd, sched


def test_the_plan_s_reads_are_its_read_holds(taught):
    world, _ = taught
    setup = RD.setup_for(world)
    sched = RD.schedule(setup, _segments(["incoming/L2/C0"]) + [
        {"name": "pick:incoming/L2/C0/push:act", "kind": "hold", "t0": 9.0, "t1": 9.5}])
    assert [(r["key"], r["read"], r["t"]) for r in sched] == [("incoming/L2/C0", "sight_far", 1.5),
                                                             ("incoming/L2/C0", "sight_near", 2.5)]
    assert {r["camera"] for r in sched} <= set(setup.cameras)


def test_reads_in_the_simulated_cell_find_their_board(taught):
    """Every read finds its own label where the cell has it (ideal images: < 0.2 mm, 0.2 deg),
    the camera's pose from the simulated arm (as the real cell has it from its joints)."""
    world, pts = taught
    rd, _ = _run(world, pts, SLOTS)
    sm = rd.summary()
    assert sm["reads"] == sm["found"] == 2 * len(SLOTS), rd.rows
    assert sm["pos_err_mm"]["max"] < 0.2 and sm["rot_err_deg"]["max"] < 0.2, sm


def test_reads_locate_displaced_racks(taught, tmp_path):
    """The racks moved (perturbed/strict_racks_moved.yaml), read at the nominal poses:
    the fit of the near reads finds the incoming rack's move, the table (not moved)
    stays where it was, and the written layout puts the jigs where the truth has them."""
    world, pts = taught
    truth = load_world(MOVED)
    rd, _ = _run(truth, pts, SLOTS)
    assert all(r["found"] for r in rd.rows), rd.rows
    # against the nominal layout: the reads are off by the move ...
    fits = RD.fit_entities(RD.setup_for(world), rd.rows)
    assert set(fits) == {"incoming", "table"}
    D_true = np.asarray(truth.entity("incoming").meta["displaced"], float)
    D = fits["incoming"]["D"]
    for f in world.features({"kind": "slot", "role": "incoming"}):
        assert np.linalg.norm((D @ f.T)[:3, 3] - truth.feature(f.key).T[:3, 3]) < 2e-4, f.key
    assert np.linalg.norm(fits["table"]["D"][:3, 3]) < 2e-4               # the table did not move
    out = RD.located_layout(STRICT, tmp_path / "located.yaml", world.name, {"incoming": fits["incoming"]})
    located = load_world(out)
    for f in located.features({"kind": "slot", "role": "incoming"}):
        assert np.linalg.norm(f.T[:3, 3] - truth.feature(f.key).T[:3, 3]) < 2e-4
    # ... and a read that does not see its board is reported, not guessed
    rows = [dict(r, found=False) if r["slot"] == "incoming/L2/C0" else r for r in rd.rows]
    assert fits["incoming"]["reads"] - 1 == RD.fit_entities(RD.setup_for(world), rows)["incoming"]["reads"]

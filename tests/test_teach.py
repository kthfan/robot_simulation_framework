"""P3 automatic teach points: the generic engine and the TM tray skills.

Fast tests teach a few slots; ``-m slow`` teaches the whole cell.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("eaik")

from rsf.core import se3  # noqa: E402
from rsf.model import registry  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402
from rsf.planning import teach as TE  # noqa: E402

CELL = Path(__file__).resolve().parents[1] / "cells" / "tm_tray"


@pytest.fixture(scope="module")
def world():
    return load_world(CELL / "layouts" / "strict.yaml")


@pytest.fixture(scope="module")
def engine(world):
    return TE.make_engine(world)


@pytest.fixture(scope="module")
def taught(world, engine):
    out = {}
    for key in ("incoming/L0/C0", "outgoing/L3/C5", "table/L0/C0"):
        for sk in ("tm_tray.pick", "tm_tray.place", "tm_tray.locate"):
            for seq in registry.get("skill", sk)(world, {"at": key}):
                out[seq.name] = (seq, engine.teach(seq))
    return out


def test_locate_reads_far_then_near_and_holds_still(taught, world):
    seq, r = taught["locate:outgoing/L3/C5"]
    assert [(p.name, p.arrive) for p in r.points] == [("sight_far", "free"), ("sight_near", "line")]
    assert [w.action for w in seq.waypoints] == ["read the AprilGrid (far)", "read the AprilGrid (near)"]
    assert all(w.meta["dwell"] == world.process["read_dwell"] > 0 for w in seq.waypoints)


def test_locate_tours_rack_by_rack_top_shelf_first():
    from rsf_plugins.tm_tray.skills import locate_order
    w = load_world(CELL / "layouts" / "strict.yaml")
    keys = [f.key for f in w.features({"kind": "slot"})]
    tour = locate_order(w, keys)
    assert sorted(tour) == sorted(keys)
    inc = [k for k in tour if k.startswith("incoming/")]
    assert inc[:5] == ["incoming/L2/C0", "incoming/L2/C1", "incoming/L2/C2", "incoming/L2/C3", "incoming/L1/C3"]


def test_every_sequence_is_taught(taught):
    bad = {n: r.problems for n, (s, r) in taught.items() if not r.ok}
    assert not bad


def test_points_reach_their_poses_and_stay_on_one_branch(taught, engine):
    for name, (seq, r) in taught.items():
        assert [p.name for p in r.points] == [w.name for w in seq.waypoints]
        for p, w in zip(r.points, seq.waypoints):
            # joints are written to 1e-6 rad, so the pose is good to a few um / urad
            dp, dr = se3.pose_error(engine.arm.fk(np.array(p.q)), w.T)
            assert dp < 5e-6 and dr < 5e-6
            assert np.all(np.array(p.q) >= engine.arm.lower) and np.all(np.array(p.q) <= engine.arm.upper)
        # inside a run (line/path arrivals) the arm keeps its configuration and moves little
        for a, b in zip(r.points, r.points[1:]):
            if b.arrive != "free":
                assert a.branch == b.branch, (name, a.name, b.name)
                assert np.abs(np.array(b.q) - np.array(a.q)).max() < 1.0


def test_pick_jaws_and_motions(taught):
    seq, r = taught["pick:incoming/L0/C0"]
    pts = {p.name: p for p in r.points}
    # no AprilGrid reads: the slot was located before the run (user 2026-11-28)
    assert [p.name for p in r.points][0] == "pre_insert"
    assert [p.arrive for p in r.points] == ["free", "line", "line", "line", "line", "path"]
    assert all(v == pytest.approx(0.057675) for v in pts["insert"].joints.values())          # open going in (D5, D30)
    assert all(v == pytest.approx(0.0413, abs=1e-4) for v in pts["withdraw"].joints.values())  # closed after lift
    assert len(pts["insert"].joints) == 2
    assert pts["push"].clearance == pytest.approx(0.0, abs=1e-6)                # touching the tray
    assert pts["insert"].clearance == pytest.approx(0.002, abs=1e-5)             # 2 mm short (push plate)
    assert pts["carry"].along_samples == 64                  # process.tilt_motion.samples (130 mm tray)


def test_tilt_motion_keeps_its_clearance(taught, world):
    """Along the tilt (withdraw -> carry, both ways), every obstacle stays at
    least process.tilt_motion.clearance (10 mm) away, the end included."""
    c = world.process["tilt_motion"]["clearance"]
    assert c == 0.010
    for name, (seq, r) in taught.items():
        if name.startswith("locate:"):                       # no tilt in a read
            continue
        tilt = next(p for p in r.points if p.arrive == "path")
        assert tilt.along_slack >= -1e-7, (name, tilt.along_slack)


def test_place_is_the_pick_reversed_where_it_should_be(taught):
    pick = {p.name: p for p in taught["pick:outgoing/L3/C5"][1].points}
    place = {p.name: p for p in taught["place:outgoing/L3/C5"][1].points}
    assert np.allclose(pick["carry"].q, place["carry"].q, atol=1e-6)
    assert np.allclose(pick["withdraw"].xyz, place["withdraw"].xyz, atol=1e-9)
    # set down 2 mm shallower than the pick lifted it after pushing 2 mm deep
    d = np.array(pick["lift"].xyz) - np.array(place["place_lift"].xyz)
    assert np.linalg.norm(d) == pytest.approx(0.004, abs=1e-6)


def test_workflow_teaches_each_motion_once(world, engine):
    from rsf.model.workflow import Workflow
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    wf.max_cycles = 2
    res = TE.teach_workflow(world, wf, engine=engine)
    every = [n for t in res["tasks"] for n in t["sequences"]]
    n_loc = len(world.features({"kind": "slot"}))
    assert all(n.startswith("locate:") for n in every[:n_loc])          # every jig located first
    names = every[n_loc:]
    assert names[:4] == ["pick:incoming/L2/C0", "place:table/L0/C0", "pick:table/L0/C0", "place:outgoing/L0/C0"]
    assert len(res["sequences"]) == len(set(every))       # the table's pick/place taught once
    assert all(r.ok for r in res["sequences"].values())
    assert [t["skill"] for t in res["tasks"][:4]] == ["tm_tray.locate", "tm_tray.transfer", "wait", "tm_tray.transfer"]
    # one arm configuration for the whole run, so free moves never flip the elbow or the wrist
    # the configuration every sequence can use (chosen automatically for this run)
    pref = tuple(res["prefer"]["branch"].values())
    sigs = {tuple(p.branch.values()) for n, r in res["sequences"].items() if "#" not in n for p in r.points}
    assert sigs == {pref}
    # the configuration changes only with the gripper empty: a sequence that ends holding the
    # tray hands over to the next in the same configuration (a taught "#" variant if need be)
    for t in res["tasks"]:
        for a, b in zip(t["sequences"], t["sequences"][1:]):
            if a.startswith("pick:"):
                assert res["sequences"][a].points[-1].branch == res["sequences"][b].points[0].branch, (a, b)
    # whole turns chosen for the run: no joint turns (nearly) a full turn between sequences
    q = np.zeros(6)
    for n in every:
        pts = res["sequences"][n].points
        assert np.abs(np.array(pts[0].q) - q).max() < math.radians(300), n
        q = np.array(pts[-1].q)


def test_align_turns_is_exact_for_recurring_sequences():
    """J6 ends at +180 deg, the shared sequence starts at -173: turn the others, not spin 353 deg."""
    import types
    arm = types.SimpleNamespace(lower=np.full(1, -math.radians(355)), upper=np.full(1, math.radians(355)))
    P = lambda *d: [types.SimpleNamespace(q=[math.radians(v)], limit_margin=0.0) for v in d]  # noqa: E731
    seqs = {"hub": types.SimpleNamespace(ok=True, points=P(-173, -180)),
            "a": types.SimpleNamespace(ok=True, points=P(175, 180)),
            "b": types.SimpleNamespace(ok=True, points=P(180, 175))}
    travel = TE.align_turns(types.SimpleNamespace(arm=arm), seqs, ["hub", "a", "b", "hub"], None)
    assert math.degrees(travel) == pytest.approx(5 + 0 + 12, abs=1e-6)   # a, b turned by -360 (else 355 + 0 + 348)
    assert math.degrees(seqs["a"].points[0].q[0]) == pytest.approx(-185)


def test_write_table(taught, tmp_path):
    results = [r for _, r in taught.values()]
    TE.write_table(results, tmp_path / "t.json", tmp_path / "t.csv", {"layout": "strict"})
    doc = json.loads((tmp_path / "t.json").read_text())
    assert doc["ok"] and len(doc["sequences"]) == len(results)
    rows = (tmp_path / "t.csv").read_text().splitlines()
    assert len(rows) == 1 + sum(len(r.points) for r in results)
    assert rows[0].startswith("sequence,point,arrive,j1_deg")


def test_a_blocked_line_is_reported(world, engine):
    """A wall across pre_insert -> insert: the sequence fails and says where."""
    from dataclasses import replace
    from rsf.model.world import Shape, Solid
    seq = registry.get("skill", "tm_tray.pick")(world, {"at": "incoming/L1/C1"})[0]
    wps = {w.name: w for w in seq.waypoints}
    mid = se3.interpolate(wps["pre_insert"].T, wps["insert"].T, 0.5)
    wall = Solid("wall", mid @ se3.make_T(None, (0.0, -0.041, 0.25)), Shape.box((0.2, 0.004, 0.004)), "", frozenset())
    blocked = []
    for w in seq.waypoints:
        if w.name == "insert":
            w = replace(w, scene=TE.Scene(w.scene.solids + (wall,), w.scene.attached))
        blocked.append(w)
    r = engine.teach(TE.Sequence(seq.name, blocked))
    assert not r.ok and any("insert" in p for p in r.problems)


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["strict", "as_built", "recommended"])
def test_whole_cell(layout):
    w = load_world(CELL / "layouts" / f"{layout}.yaml")
    res = TE.teach_targets(w, ["tm_tray.pick", "tm_tray.place"])
    assert len(res) == 74 and all(r.ok for r in res), [(r.sequence, r.problems) for r in res if not r.ok]

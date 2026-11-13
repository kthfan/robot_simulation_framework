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
        for sk in ("tm_tray.pick", "tm_tray.place"):
            for seq in registry.get("skill", sk)(world, {"at": key}):
                out[seq.name] = (seq, engine.teach(seq))
    return out


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
    assert [p.arrive for p in r.points] == ["free", "line", "free", "line", "line", "line", "line", "path"]
    assert all(v == pytest.approx(0.057675) for v in pts["insert"].joints.values())          # open going in (D5, D30)
    assert all(v == pytest.approx(0.0413, abs=1e-4) for v in pts["withdraw"].joints.values())  # closed after lift
    assert len(pts["insert"].joints) == 2
    assert pts["push"].clearance == pytest.approx(0.0, abs=1e-6)                # touching the tray
    assert pts["insert"].clearance == pytest.approx(0.002, abs=1e-5)             # 2 mm short (push plate)
    assert pts["carry"].along_samples == 16


def test_tilt_motion_keeps_its_clearance(taught, world):
    """Along the tilt (withdraw -> carry, both ways), every obstacle stays at
    least process.tilt_motion.clearance (10 mm) away, the end included."""
    c = world.process["tilt_motion"]["clearance"]
    assert c == 0.010
    for name, (seq, r) in taught.items():
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
    names = [n for t in res["tasks"] for n in t["sequences"]]
    assert names[:4] == ["pick:incoming/L2/C0", "place:table/L0/C0", "pick:table/L0/C0", "place:outgoing/L0/C0"]
    assert len(res["sequences"]) == len(set(names))       # the table's pick/place taught once
    assert all(r.ok for r in res["sequences"].values())
    assert [t["skill"] for t in res["tasks"][:3]] == ["tm_tray.transfer", "wait", "tm_tray.transfer"]


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

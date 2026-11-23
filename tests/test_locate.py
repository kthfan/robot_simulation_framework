"""P6 locating: racks displaced in the layout (the truth a run meets), and located
again from AprilGrid reads (tools/locate_slots.py)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from rsf.model import registry
from rsf.model.layout import build_world, displacement, load_document, load_world

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "cells" / "tm_tray"
STRICT = CELL / "layouts" / "strict.yaml"
MOVED = CELL / "layouts" / "perturbed" / "strict_racks_moved.yaml"
sys.path.insert(0, str(ROOT / "tools"))


def test_displacement_spec():
    """Rotate about a point, then translate; or a matrix as given."""
    D = displacement({"xyz": [0.01, 0.0, 0.0], "rpy": [0.0, 0.0, math.pi / 2], "about": [1.0, 0.0, 0.0]})
    assert D[:3, 3] @ np.ones(3) == pytest.approx(D[:3, 3].sum())
    assert (D @ [1.0, 0.0, 0.0, 1.0])[:3] == pytest.approx([1.01, 0.0, 0.0])      # the pivot only translates
    assert (D @ [2.0, 0.0, 0.0, 1.0])[:3] == pytest.approx([1.01, 1.0, 0.0])
    M = np.eye(4)
    M[:3, 3] = [1, 2, 3]
    assert displacement({"matrix": M.tolist()}) == pytest.approx(M)
    assert displacement(None) == pytest.approx(np.eye(4))


def test_displaced_racks_move_rigidly():
    """Slots, the jigs and trays placed on them, the labels and every pick pose move with the rack."""
    nom, moved = load_world(STRICT), load_world(MOVED)          # loads the plugin
    import rsf_plugins.tm_tray.sequence as SEQ
    from rsf_plugins.tm_tray.geometry import slot_of
    from rsf_plugins.tm_tray.tool import tool_geometry
    D = {e.name: np.asarray(e.meta["displaced"]) for e in moved.entities if "displaced" in e.meta}
    assert set(D) == {"incoming", "outgoing"}
    labels0 = {l["key"]: l for l in registry.get("vision", "tm_tray.labels")(nom)}
    labels1 = {l["key"]: l for l in registry.get("vision", "tm_tray.labels")(moved)}
    for f in nom.features({"kind": "slot"}):
        g = moved.feature(f.key)
        Df = D.get(f.entity, np.eye(4))
        assert g.T == pytest.approx(Df @ f.T, abs=1e-12)
        assert labels1[f.key]["board_T"] == pytest.approx(Df @ labels0[f.key]["board_T"], abs=1e-12)
        if f.entity in D:
            for a, b in zip(SEQ.pick_phases(nom.process, slot_of(f), tool_geometry(nom)),
                            SEQ.pick_phases(moved.process, slot_of(g), tool_geometry(moved))):
                assert np.asarray(b.T) == pytest.approx(Df @ np.asarray(a.T), abs=1e-12), (f.key, a.name)
    for o0 in nom.objects:
        o1 = next(o for o in moved.objects if o.name == o0.name)
        assert o1.T == pytest.approx(D.get(nom.feature(o0.at).entity, np.eye(4)) @ o0.T, abs=1e-12)
    jig = "jig/incoming/L0/C0"
    assert moved.entity(jig).solids[0].T == pytest.approx(D["incoming"] @ nom.entity(jig).solids[0].T, abs=1e-12)


def test_slot_at_inverts_jig_pose():
    w = load_world(STRICT)
    from rsf_plugins.tm_tray.geometry import jig_pose, slot_at, slot_of
    for f in w.features({"kind": "slot"}):
        s = slot_of(f)
        back = slot_at(s, jig_pose(s))
        for k in ("lateral", "label_plane_dist", "height"):
            assert getattr(back, k) == pytest.approx(getattr(s, k), abs=1e-12)
        assert back.approach == pytest.approx(s.approach, abs=1e-12)
    T = jig_pose(s)
    T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(0.01), -math.sin(0.01)], [0, math.sin(0.01), math.cos(0.01)]])
    with pytest.raises(ValueError, match="not level"):
        slot_at(s, T)


def test_displace_needs_a_known_key():
    doc = load_document(STRICT)
    doc["entities"]["incoming"]["displace"] = {"xyz": [0, 0, 0], "yaw": 1}
    with pytest.raises(ValueError, match="displace keys"):
        build_world(doc)


def test_fit_planar_recovers_a_floor_move():
    from locate_slots import fit_planar
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, (3, 20))
    th = math.radians(0.7)
    D = np.eye(4)
    D[:3, :3] = [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]
    D[:3, 3] = [0.004, -0.006, 0.002]
    Q = (D @ np.vstack([P, np.ones(20)]))[:3]
    assert fit_planar(P, Q) == pytest.approx(D, abs=1e-12)


def test_locate_strict_racks(tmp_path):
    """Both racks of the strict scenario located from simulated reads: every jig within 0.05 mm."""
    pytest.importorskip("aprilgrid_detector")
    pytest.importorskip("mujoco")
    import locate_slots
    out = tmp_path / "located.yaml"
    rc = locate_slots.main(["--truth", str(MOVED), "--nominal", str(STRICT), "--teach", str(CELL / "teach" / "strict.json"),
                            "--racks", "incoming", "outgoing", "--out-layout", str(out), "--out", str(tmp_path / "r.json")])
    assert rc == 0
    rep = json.loads((tmp_path / "r.json").read_text())["summary"]
    assert rep["located"] == rep["slots"] == 36
    for rack, fit in rep["racks"].items():
        assert fit["jig_error_mm"]["max"] < 0.05 and fit["jig_error_deg"]["max"] < 0.01, rack
    located, truth = load_world(out), load_world(MOVED)
    for f in truth.features({"kind": "slot"}):
        assert np.abs(located.feature(f.key).T[:3, 3] - f.T[:3, 3]).max() < 5e-5

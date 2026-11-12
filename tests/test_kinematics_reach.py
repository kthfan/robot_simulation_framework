"""EAIK kinematics backend and the reach sweep, against the original project.

Needs the EAIK fork and robot_arm_motion_planner (``pip install .[eaik]``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("eaik")
pytest.importorskip("motion_planner.robust_eaik")

from rsf.core import se3  # noqa: E402
from rsf.kinematics import load_arm  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402
from rsf.model.urdf import UrdfModel  # noqa: E402
from rsf.planning import reach  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ROOT / "cells" / "tm_tray" / "layouts"
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference" / "ai_robotics_v2"))
import cell_core as REF  # noqa: E402
from cell_core import sequence as RS  # noqa: E402

USD_FLANGE_R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@pytest.fixture(scope="module")
def worlds():
    return {p: load_world(LAYOUTS / f"{p}.yaml") for p in ("as_built", "recommended", "strict")}


@pytest.fixture(scope="module")
def arm(worlds):
    return load_arm(worlds["strict"])


def test_flange_uses_the_usd_convention(arm):
    """cell_core builds every target in tm5s.usd's flange convention (D21); the
    assembled URDF's flange is that frame at q = 0, so no conversion is needed."""
    assert np.allclose(arm.backend.fk(np.zeros(6))[:3, :3], USD_FLANGE_R0, atol=1e-6)


def test_eaik_fk_equals_the_urdf(arm, worlds):
    model = UrdfModel.from_file(worlds["strict"].robot.urdf)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(arm.lower / 2, arm.upper / 2)
        T = worlds["strict"].robot.base @ model.fk(dict(zip(arm.joint_names, q)))["flange"]
        assert max(se3.pose_error(arm.fk(q), T)) < 1e-12


def test_ik_finds_the_configuration_it_came_from(arm):
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(arm.lower / 2, arm.upper / 2)
        sols = arm.ik(arm.fk(q), expand_turns=False)
        assert sols and all(max(se3.pose_error(arm.fk(s), arm.fk(q))) < 1e-8 for s in sols)
        wrapped = [np.mod(s - q + np.pi, 2 * np.pi) - np.pi for s in sols]
        assert any(np.abs(w).max() < 1e-6 for w in wrapped)


NO_AIR = str((ROOT / "cells" / "tm_tray" / "assets" / "tm5s_egu50_no_air_cylinder.urdf").resolve())


@pytest.mark.parametrize("preset", ["as_built", "recommended", "strict"])
def test_grasp_targets_equal_the_original_sequence(preset):
    """With the D29 gripper (no push plate) the targets are cell_core's."""
    from rsf.model import registry
    cfg = REF.load_preset(preset)
    world = load_world(LAYOUTS / f"{preset}.yaml", {"robot.urdf": NO_AIR, "process.carry_tilt_deg": 0.0})
    fn = registry.get("targets", "tm_tray.grasp")
    for s in REF.enumerate_slots(cfg):
        mine = fn(world, world.feature(s.key))
        ref = RS.grasp_sequence(cfg, s)
        assert list(mine) == list(ref)
        for p in ref:
            assert np.allclose(mine[p], ref[p], atol=1e-12), (s.key, p)


@pytest.mark.parametrize("preset", ["as_built", "recommended", "strict"])
def test_every_slot_is_reachable_with_the_current_sequence(worlds, preset):
    """Kinematically, every pose of the pick and place (push plate, 15 deg tilt)
    is reachable at every slot on all three mounts, as_built included."""
    world = worlds[preset]
    s = reach.summary(reach.sweep(world, load_arm(world), "tm_tray.cycle"))
    assert s["reachable"] == 37 and s["poses_reachable"] == s["poses"] == 37 * 13, s["unreachable"]


def test_pre_d28_sequence_reproduces_findings_04(worlds):
    """With the sequence findings/04 swept (insert to the tray back margin, lift
    40 mm), as_built loses exactly outgoing/L3/C0 and /C5 at insert and lift.
    This pins the IK against the original project's published result."""
    cfg = REF.load_preset("as_built")
    arm = load_arm(worlds["as_built"])
    bad = {}
    for s in REF.enumerate_slots(cfg):
        poses = RS.grasp_sequence(cfg, s)
        depth = RS.A.TRAY_BACK_DEPTH - RS.BACK_MARGIN
        poses["insert"] = RS._pose(s, s.label_plane_dist + RS.A.LABEL_PROUD_OF_FACE + depth - RS.A.PLATE_FZ_TIP,
                                   RS.insertion_flange_z(s))
        poses["lift"] = poses["insert"].copy()
        poses["lift"][2, 3] += 0.040
        poses["withdraw"] = RS.pre_insert_pose(s)
        poses["withdraw"][2, 3] += 0.040
        miss = [p for p, T in poses.items() if not arm.ik(T, expand_turns=False)]
        if miss:
            bad[s.key] = miss
    assert bad == {"outgoing/L3/C0": ["insert", "lift"], "outgoing/L3/C5": ["insert", "lift"]}


def test_pick_with_the_push_plate(worlds):
    """With the push plate the pick is cell_core's grasp except: insert stops 2 mm
    before a tray placed 2 mm shallow, a push follows to 2 mm past the seat, the
    tray is lifted from there, and the tilted carry backs out first."""
    from rsf.model import registry
    import rsf_plugins.tm_tray.sequence as SEQ
    from rsf_plugins.tm_tray.tool import tool_geometry
    cfg = REF.load_preset("strict")
    world = worlds["strict"]
    tool = tool_geometry(world)
    d = SEQ.depths(world.process, tool)
    K = 0.321 - tool["push_front_z"]
    f0 = RS.A.TRAY_FRONT_DEPTH
    assert d["insert"] == pytest.approx(f0 - 0.004 + K, abs=1e-12)
    assert d["push"] == pytest.approx(f0 + 0.002 + K, abs=1e-12)
    assert d["place"] == pytest.approx(f0 - 0.002 + K, abs=1e-12)
    fn = registry.get("targets", "tm_tray.grasp")
    for s in REF.enumerate_slots(cfg)[::5]:
        mine, ref = fn(world, world.feature(s.key)), RS.grasp_sequence(cfg, s)
        assert list(mine) == ["sight_far", "sight_near", "pre_insert", "insert", "push", "lift", "withdraw", "carry"]
        a = s.approach_vec / np.linalg.norm(s.approach_vec)
        for p in ("sight_far", "sight_near", "pre_insert", "withdraw"):
            assert np.allclose(mine[p], ref[p], atol=1e-12), (s.key, p)
        for p, depth in (("insert", d["insert"]), ("push", d["push"])):
            assert np.allclose(mine[p][:3, 3] - ref["insert"][:3, 3], (depth - RS.insert_tip_depth()) * a, atol=1e-12)
        assert np.allclose(mine["lift"][:3, 3] - mine["push"][:3, 3], [0, 0, RS.LIFT_HEIGHT], atol=1e-12)
        tilted = RS.carry_pose(s, 15.0)
        assert np.allclose(mine["carry"][:3, :3], tilted[:3, :3], atol=1e-12)
        assert np.allclose(mine["carry"][:3, 3], tilted[:3, 3] - SEQ.tilt_retreat(tool["push_front_z"] + RS.A.TRAY_Y / 2) * a,
                           atol=1e-12)

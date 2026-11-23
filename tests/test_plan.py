"""P4: a workflow run as one timed, CPU-checked trajectory.

Free moves come from cuRobo on a GPU; here they are straight joint lines
(``--paths linear``), which the CPU gate is expected to reject where they clip
something. Everything else (constrained lines, the tilt, jaw holds, timing)
must pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("eaik")

from rsf.model.layout import load_world  # noqa: E402
from rsf.model.workflow import Workflow  # noqa: E402
from rsf.planning import plan as PL  # noqa: E402
from rsf.planning import timing as TM  # noqa: E402
from rsf.planning.teach import make_engine  # noqa: E402

CELL = Path(__file__).resolve().parents[1] / "cells" / "tm_tray"


@pytest.fixture(scope="module")
def run():
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    wf.max_cycles = 1
    doc = json.loads((CELL / "teach" / "strict_inspect_and_sort.json").read_text())
    teach = {s["sequence"]: s["points"] for s in doc["sequences"]}
    segs = PL.segments_for_workflow(world, wf, teach)
    for s in segs:
        if s.kind == "free":
            s.Q = np.linspace(s.q_start, s.q_goal, 50)
    eng = make_engine(world)
    traj, problems = PL.assemble(world, eng.arm, eng.checker, segs)
    return world, segs, traj, problems, eng


def test_segments_of_one_cycle(run):
    world, segs, traj, problems, eng = run
    kinds = [s.kind for s in segs]
    assert kinds.count("free") == 7          # home->pick, 2 inside picks, 2 carries, pick table, ->home
    assert kinds.count("path") == 4          # the tilt, twice per transfer
    names = [s.name for s in segs]
    assert names[0].startswith("free001:pick:incoming/L2/C0/sight_far") and names[-1].endswith(":home")
    carried = [s for s in segs if s.kind == "free" and s.carried is not None]
    assert len(carried) == 2 and all(s.caps == "carry" for s in carried)


def test_only_free_straight_lines_can_fail(run):
    world, segs, traj, problems, eng = run
    assert all(p.startswith("free") for p in problems), problems


def test_trajectory_is_continuous_and_complete(run):
    world, segs, traj, problems, eng = run
    assert traj.joint_names[:6] == [f"joint_{i}" for i in range(1, 7)]
    assert traj.q.shape == (len(traj.t), 8)
    assert np.all(np.diff(traj.t) > 0)
    dq = np.abs(np.diff(traj.q[:, :6], axis=0)).max()
    assert dq < 0.05                          # no jumps between samples (dt 0.01 s)
    assert np.allclose(traj.q[0, :6], 0.0) and np.allclose(traj.q[-1, :6], 0.0)   # home -> home
    lo, hi = eng.arm.lower, eng.arm.upper
    assert np.all(traj.q[:, :6] >= lo - 1e-9) and np.all(traj.q[:, :6] <= hi + 1e-9)


def test_constrained_lines_keep_their_caps(run):
    """Inside the shelf the flange keeps D30's IN_JIG caps (0.05 m/s, 0.25 m/s^2)."""
    world, segs, traj, problems, eng = run
    caps = TM.caps_from(world.process["motion"]["caps"]["in_jig"])
    for s in segs:
        if s.kind in ("line", "path") and s.caps == "in_jig" and len(s.t) > 3:
            P = np.array([eng.arm.fk(q)[:3, 3] for q in s.Q])
            k = TM.point_kinematics(s.t, P)
            assert k["v"] <= caps.v * 1.05, (s.name, k)
            assert k["a"] <= caps.a * 1.25, (s.name, k)


def test_jaws_close_after_lift_and_open_before_setting_down(run):
    world, segs, traj, problems, eng = run
    j = traj.joint_names.index("right_gripper_joint")
    seg = {s["name"]: s for s in traj.segments}
    t_lift = seg["pick:incoming/L2/C0/lift:act"]
    i0 = np.searchsorted(traj.t, t_lift["t0"])
    i1 = np.searchsorted(traj.t, t_lift["t1"])
    assert traj.q[i0, j] == pytest.approx(0.057675, abs=1e-6)
    assert traj.q[i1, j] == pytest.approx(0.0413, abs=1e-4)
    assert t_lift["t1"] - t_lift["t0"] == pytest.approx((0.057675 - 0.0413) / 0.05 + 0.5, abs=0.02)


def test_job_round_trip(run, tmp_path):
    world, segs, traj, problems, eng = run
    job = PL.write_job(world, segs, tmp_path / "job.json", root=CELL.parents[1], arm=eng.arm)
    assert job["robot_urdf"] == "cells/tm_tray/assets/tm5s_egu50.urdf"
    assert len(job["segments"]) == 7 and job["limit_margin_deg"] == 5.0
    carried = [s for s in job["segments"] if s["carried"]]
    assert len(carried) == 2 and len(carried[0]["carried"]["boxes"]) == 3
    # the carry rule for cuRobo: a tool axis within a cone of the vertical, the largest cone
    # inside both rules (15 deg from level, no outward pull: ~6 deg forward of level)
    b = carried[0]["tool_bound"]
    assert b["reference_axis"] == [0.0, 0.0, 1.0]
    assert math.radians(8.0) < b["max_angle"] < math.radians(15.0)
    s0 = next(x for x in segs if x.name == carried[0]["name"])
    frame = PL.carry_frame(eng.arm, s0, **PL.carry_rule(PL._motion(world.process)))
    axis = np.array(b["tool_axis"])
    up0 = -(eng.arm.fk(s0.q_start)[:3, :3].T @ np.array([0.0, 0.0, -1.0]))   # tool-frame up at the start
    assert math.acos(min(1.0, float(axis @ up0))) <= b["max_angle"]             # the start is inside
    for k in range(36):                                                          # the cone's edge keeps both rules
        e = _cone_edge(axis, b["max_angle"], 2 * math.pi * k / 36)
        g = -e
        assert float(g @ frame[0]) >= frame[2] - 1e-9 and float(g @ frame[1]) <= 1e-9
    assert all(s["tool_bound"] is None for s in job["segments"] if not s["carried"])
    assert len(job["scenes"]) < len(job["segments"])
    assert all("pedestal/plate" not in job["scenes"][s["scene"]]["cuboid"] for s in job["segments"])  # base mount
    fake = {s["name"]: {"success": True, "q": [s["q_start"], s["q_goal"]], "dt": 0.1} for s in job["segments"]}
    fake[job["segments"][0]["name"]]["success"] = False
    assert PL.attach_results(segs, fake) == [job["segments"][0]["name"]]


def _cone_edge(axis, angle, phi):
    a = np.asarray(axis, float)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return math.cos(angle) * a + math.sin(angle) * (math.cos(phi) * u + math.sin(phi) * v)


def test_carry_bound_geometry():
    """The cuRobo cone: the whole tilt cone when the outward plane is farther than its
    edge; else the largest cone between the two, touching both."""
    g = np.array([0.0, 0.0, -1.0])
    c = np.array([0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])   # plane 30 deg away
    axis, ang = PL.carry_bound((g, c, math.cos(math.radians(15.0))))
    assert ang == pytest.approx(math.radians(15.0)) and axis == pytest.approx(-g)
    c = np.array([0.0, math.cos(math.radians(6.0)), math.sin(math.radians(6.0))])     # plane 6 deg away
    delta = math.asin(-float(g @ c))
    assert delta == pytest.approx(math.radians(6.0))
    axis, ang = PL.carry_bound((g, c, math.cos(math.radians(15.0))))
    assert ang == pytest.approx(0.5 * (math.radians(15.0) + delta))
    for k in range(24):
        e = -_cone_edge(axis, ang, 2 * math.pi * k / 24)                   # gravity on the cone's edge
        assert float(e @ g) >= math.cos(math.radians(15.0)) - 1e-9 and float(e @ c) <= 1e-9


def test_carry_orientation(run):
    """A free carry may not roll the tray off the plate: J6 +60 deg rolls it."""
    world, segs, traj, problems, eng = run
    s = next(x for x in segs if x.kind == "free" and x.carried is not None)
    # measured from level (the cell's rule): the start (5 deg back) is fine, a 20 deg roll is not
    lv = PL.carry_frame(eng.arm, s, 15.0, "level")
    assert PL.carry_violation(eng.arm, s.q_start, lv) is None
    q = s.q_start.copy()
    q[5] += math.radians(20.0)
    assert "tilted" in PL.carry_violation(eng.arm, q, lv)
    q[5] -= math.radians(12.0)                                            # 8 deg roll + 5 back: < 15 from level
    assert PL.carry_violation(eng.arm, q, lv) is None
    # measured from the carry's start
    frame = PL.carry_frame(eng.arm, s, 30.0, "start")
    assert PL.carry_violation(eng.arm, s.q_start, frame) is None
    assert PL.carry_violation(eng.arm, s.q_goal, frame) is None          # set down the way it was picked
    q = s.q_start.copy()
    q[5] += math.radians(60.0)
    assert "tilted" in PL.carry_violation(eng.arm, q, frame)
    q[5] -= math.radians(45.0)                                            # 15 deg: fine
    assert PL.carry_violation(eng.arm, q, frame) is None


def test_carried_object_against_the_arm(run):
    """The gate checks a carried object against the arm's own links (not those holding it)."""
    from rsf.model.world import Shape, Solid
    from rsf.core import se3
    world, segs, traj, problems, eng = run
    c = eng.checker
    c.set_obstacles([])
    for n in list(c.attached):
        c.detach(n)
    q = dict(zip(eng.arm.joint_names, [0.0, 0.0, 1.5, 0.0, 1.5, 0.0]))
    box = Solid("blob", np.eye(4), Shape.box((0.2, 0.2, 0.2)), "", frozenset())   # crosses a link's surface
    c.attach("held", "flange", [box], se3.make_T(None, (0.0, 0.0, 0.45)))     # in front of the gripper: clear
    assert c.check(q).ok
    c.detach("held")
    poses = c.link_poses(q)
    at_link4 = np.linalg.inv(poses["flange"]) @ poses["link_4"]
    c.attach("held", "flange", [box], at_link4)                                # inside the forearm's wrist
    r = c.check(q)
    assert not r.ok and r.self_violations[0].rule == "carried vs robot"
    assert r.self_violations[0].obstacle.startswith("link_")
    c.detach("held")

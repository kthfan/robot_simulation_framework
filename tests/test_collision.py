"""The CPU collision backend: distances, margins, contact rules, attachments, self pairs.

The probes are AI-Robotics-V2 check_cell's live-mesh self-tests (``PROBES``),
placed in the flange frame of the assembled robot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")

from rsf.collision import ContactPolicy, ContactRule  # noqa: E402
from rsf.collision.coal_backend import CoalChecker, candidate_self_pairs, rigid_bodies  # noqa: E402
from rsf.core import se3  # noqa: E402
from rsf.model.urdf import UrdfModel  # noqa: E402
from rsf.model.world import Shape, Solid  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "cells" / "tm_tray" / "assets"
JAW_OPEN = 0.057675          # cell_core JAW_Q_OPEN
JAW_X_IN, JAW_X_OUT = 0.02565, 0.0310
JAWS = ("right_gripper_joint", "left_gripper_joint")


@pytest.fixture(scope="module")
def robot():
    return UrdfModel.from_file(ASSETS / "tm5s_egu50.urdf")


def q_of(robot, q6=(0.3, -0.4, 1.2, 0.2, 1.4, 0.5), jaw=JAW_OPEN):
    d = {f"joint_{i + 1}": v for i, v in enumerate(q6)}
    d.update({j: jaw for j in JAWS})
    return d


def probe(robot, q, centre, size, name="probe"):
    """A box given in the flange frame, as a world solid."""
    T = robot.fk(q)["flange"] @ se3.make_T(None, centre)
    return Solid(name, T, Shape.box(size), "probe", frozenset({"probe"}))


@pytest.mark.parametrize("centre,size,margin,hit", [
    ((0.0, -0.047, 0.2), (0.01, 0.004, 0.01), 0.0, True),     # straddles the plate underside
    ((0.0, -0.051, 0.2), (0.01, 0.002, 0.01), 0.0, False),    # 4 mm below the plate
    ((0.0, -0.051, 0.2), (0.01, 0.002, 0.01), 0.005, True),   # ...inside a 5 mm margin
    (((JAW_X_IN + JAW_X_OUT) / 2 + JAW_OPEN, 0.0, 0.16), (0.008, 0.04, 0.02), 0.0, True),  # across a jaw
])
def test_probes(robot, centre, size, margin, hit):
    q = q_of(robot)
    ck = CoalChecker(robot, np.eye(4), ContactPolicy(margin))
    ck.set_obstacles([probe(robot, q, centre, size)])
    assert (not ck.check(q, self_collision=False).ok) == hit


def test_allow_rule_lets_the_jaws_touch(robot):
    q = q_of(robot)
    jaw_probe = probe(robot, q, ((JAW_X_IN + JAW_X_OUT) / 2 + JAW_OPEN, 0.0, 0.16), (0.008, 0.04, 0.02))
    pol = ContactPolicy(0.0, [ContactRule(allow=True, links=("right_gripper", "left_gripper"), tags=("probe",))])
    ck = CoalChecker(robot, np.eye(4), pol)
    ck.set_obstacles([jaw_probe])
    assert ck.check(q, self_collision=False).ok


def test_signed_distance_and_margin_rule(robot):
    q = q_of(robot)
    below = probe(robot, q, (0.0, -0.056, 0.2), (0.01, 0.002, 0.01))     # top face 9 mm under the plate
    ck = CoalChecker(robot, np.eye(4), ContactPolicy(0.005))
    ck.set_obstacles([below])
    r = ck.check(q, self_collision=False)
    assert r.ok and r.worst.distance == pytest.approx(0.009, abs=1e-6)
    ck.policy = ContactPolicy(0.0, [ContactRule(margin=0.010, tags=("probe",))])
    ck._rule_cache.clear()
    r = ck.check(q, self_collision=False)
    assert not r.ok and r.violations[0].slack == pytest.approx(-0.001, abs=1e-6)


def test_attached_object_moves_with_the_link(robot):
    q = q_of(robot)
    box = Solid("cargo", np.eye(4), Shape.box((0.05, 0.05, 0.05)))
    T_flange = robot.fk(q)["flange"]
    wall = Solid("wall", T_flange @ se3.make_T(None, (0.0, 0.0, 0.6)), Shape.box((0.3, 0.3, 0.01)))
    ck = CoalChecker(robot, np.eye(4), ContactPolicy(0.0))
    ck.set_obstacles([wall])
    assert ck.check(q, self_collision=False).ok
    ck.attach("cargo", "flange", [box], se3.make_T(None, (0.0, 0.0, 0.6)))   # into the wall
    r = ck.check(q, self_collision=False)
    assert not r.ok and r.violations[0].link == "attached/cargo"
    ck.detach("cargo")
    assert ck.check(q, self_collision=False).ok


def test_robot_base_moves_everything(robot):
    q = q_of(robot)
    base = se3.make_T(se3.rot_z(np.pi), (0.0, 0.0, 0.735))
    p = probe(robot, q, (0.0, -0.047, 0.2), (0.01, 0.004, 0.01))
    moved = Solid(p.name, base @ p.T, p.shape)
    ck = CoalChecker(robot, base, ContactPolicy(0.0))
    ck.set_obstacles([moved])
    assert not ck.check(q, self_collision=False).ok
    ck.set_obstacles([p])                     # the same box, not moved with the base
    assert ck.check(q, self_collision=False).ok


def test_self_pairs_skip_rigid_and_adjacent_bodies(robot):
    body = rigid_bodies(robot)
    assert body["gripper_body"] == body["flange"] == body["link_6"]
    pairs = {frozenset(p) for p in candidate_self_pairs(robot)}
    assert frozenset(("link_5", "link_6")) not in pairs          # adjacent
    assert frozenset(("link_6", "gripper_body")) not in pairs    # one rigid body
    assert frozenset(("link_3", "gripper_body")) in pairs


def test_self_collision_is_found(robot):
    """Fold the wrist so the plate sweeps through the forearm."""
    ck = CoalChecker(robot, np.eye(4), ContactPolicy(0.0))
    rng = np.random.default_rng(3)
    names = [f"joint_{i}" for i in range(1, 7)]
    lo, hi = robot.limits(names)
    hits = 0
    for _ in range(60):
        q = dict(zip(names, rng.uniform(lo, hi)))
        q.update({j: JAW_OPEN for j in JAWS})
        hits += bool(ck.check(q).self_violations)
    assert 0 < hits < 60

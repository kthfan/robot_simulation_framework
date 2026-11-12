"""M3: is every slot's grasp sequence collision-free in the cell? (CPU port)

Port of AI-Robotics-V2 ``program/collision/check_cell.py`` onto the framework:
EAIK branches (``rsf.kinematics``), exact meshes on the CPU
(``rsf.collision.coal_backend``) instead of cuRobo, and contact rules instead of
three checkers and one world per obstacle group. Semantics kept (D28, D29):

* every exact, in-limit IK branch of every grasp phase is a candidate;
* everything keeps ``MARGIN`` except the target jig and target tray (margin 0,
  the plate goes under the tray); the jaws may touch the target tray (they clamp
  its sides); the arm base rests on its mount plate;
* from ``lift`` on the target tray leaves its jig and rides on the plate,
  attached to the flange; it must keep ``MARGIN`` from everything but its own
  jig (margin 0);
* a slot passes when ``pre_insert -> insert -> lift`` have one continuous branch
  chain and every other phase has at least one passing branch;
* self-collision: exact link meshes, non-adjacent bodies (cuRobo used spheres).

The jaws are at the open command throughout (``JAW_Q_OPEN``), as in M3.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from rsf.collision import ContactPolicy, ContactRule
from rsf.collision.coal_backend import CoalChecker
from rsf.planning.chain import ContinuityRule, best_chain
from rsf.model.urdf import UrdfModel

from . import frames as F
from . import hardware as A
from . import sequence as SEQ
from .geometry import slot_of
from .tool import tool_geometry

MARGIN = 0.005
CHAIN = ("pre_insert", "insert", "lift")
CARRIED_PHASES = ("lift", "withdraw", "carry")
JAWS = ("right_gripper_joint", "left_gripper_joint")
JAW_LINKS = ("right_gripper", "left_gripper")
CONTINUITY = ContinuityRule(sign_joints=(2, 4), max_step=1.0)


def carried_tray_pose(tip_depth: Optional[float] = None) -> np.ndarray:
    """Tray frame (rails' underside centre; x across, y along, z up) in the flange
    frame while it rides on the plate: recess on the plate's upper face, at the
    depth :func:`sequence.insert_pose` picked it up (cell_core.scene.carried_tray_box)."""
    R = F.USD_FLANGE_R0.T        # tray x -> flange x, tray y -> flange -z, tray z -> flange y
    depth = SEQ.insert_tip_depth() if tip_depth is None else tip_depth
    p = (0.0, A.PLATE_FY_TOP - A.TRAY_RECESS, A.PLATE_FZ_TIP + A.JIG_Y_OFFSET - depth)
    return F.make_pose(R, p)


def policy(key: str, *, carried: bool) -> ContactPolicy:
    rules = [
        ContactRule(allow=True, links=JAW_LINKS, names=(f"tray/{key}/*",), label="jaws clamp the target tray"),
        ContactRule(margin=0.0, names=(f"tray/{key}/*",), label="target tray"),
        ContactRule(margin=0.0, names=(f"jig/{key}/*",), label="target jig"),
        ContactRule(allow=True, links=("link_0",), tags=("robot_seat",), label="base on its mount"),
    ]
    return ContactPolicy(MARGIN, rules)


def group_of(name: str, key: str) -> str:
    """The original report's obstacle groups."""
    if name.startswith(f"jig/{key}/"):
        return "target/jig"
    if name.startswith(f"tray/{key}/"):
        return "target/tray"
    head = name.split("/")
    if head[0] in ("incoming", "outgoing"):
        return f"rack/{head[0]}/" + ("shelves" if head[1].startswith("shelf") else "legs")
    return {"jig": "jigs", "tray": "trays"}.get(head[0], head[0])


class CellChecker:
    """One coal checker for the world's robot, re-pointed per slot and phase."""

    def __init__(self, world, arm, *, jaw: float = A.JAW_Q_OPEN):
        self.world, self.arm, self.jaw = world, arm, jaw
        self.model = UrdfModel.from_file(world.robot.urdf)
        self.checker = CoalChecker(self.model, world.robot.base, ContactPolicy(MARGIN), cutoff=0.06)
        self.tool = tool_geometry(world)

    def _q(self, q6) -> dict:
        d = dict(zip(self.arm.joint_names, np.asarray(q6, float)))
        d.update({j: self.jaw for j in JAWS})
        return d

    def judge(self, key: str, phase: str, q6) -> dict:
        carried = phase in CARRIED_PHASES
        self.checker.policy = policy(key, carried=carried)
        self.checker._rule_cache.clear()
        r = self.checker.check(self._q(q6), group=lambda n: group_of(n, key))
        rest = [p.slack + MARGIN for g, p in r.nearest_by_group.items() if not g.startswith("target/")]
        score = min(rest) if rest else 0.06
        return {"q": [round(float(v), 6) for v in q6], "ok": r.ok,
                "score": score,
                "worst": None if r.worst is None else
                {"link": r.worst.link, "obstacle": r.worst.obstacle, "distance": round(r.worst.distance, 5),
                 "margin": r.worst.margin, "rule": r.worst.rule},
                "violations": [(v.link, v.obstacle, round(v.distance, 5), v.margin) for v in r.violations[:5]],
                "self": [(v.link, v.obstacle, round(v.distance, 5)) for v in r.self_violations[:5]],
                "groups": {g: round(p.distance, 5) for g, p in sorted(r.nearest_by_group.items())}}

    def _world_for(self, key: str, carried: bool):
        tray = self.world.object_at(key)
        if carried and tray is not None:
            solids = self.world.solids(exclude_objects=[tray.name])
            self.checker.set_obstacles(solids)
            depth = SEQ.insert_tip_depth(**SEQ.push_kwargs(self.world.process, self.tool))
            self.checker.attach("tray", "flange", tray.solids, carried_tray_pose(depth))
        else:
            self.checker.set_obstacles(self.world.solids())
            self.checker.detach("tray")

    def check_slot(self, feature) -> dict:
        key = feature.key
        poses = SEQ.grasp_sequence(self.world.process, slot_of(feature), tool=self.tool)
        phases = {}
        for carried in (False, True):
            names = [p for p in SEQ.GRASP_PHASES if (p in CARRIED_PHASES) == carried]
            self._world_for(key, carried)
            for p in names:
                phases[p] = [self.judge(key, p, q) for q in self.arm.ik(poses[p], expand_turns=False)]
        self.checker.detach("tray")
        missing = [p for p in SEQ.GRASP_PHASES if not any(b["ok"] for b in phases[p])]
        idx, qs = best_chain([phases[p] for p in CHAIN], self.arm.lower, self.arm.upper, CONTINUITY)
        return {"slot": key, "ok": idx is not None and not missing, "missing": missing,
                "chain": None if idx is None else list(idx),
                "chain_q": None if qs is None else [[round(float(v), 6) for v in q] for q in qs],
                "phases": phases}


def check_cell(world, arm=None, select=None) -> dict:
    from rsf.kinematics import load_arm
    arm = arm or load_arm(world)
    cc = CellChecker(world, arm)
    slots = [cc.check_slot(f) for f in world.features(select or {"kind": "slot"})]
    n_ok = sum(s["ok"] for s in slots)
    return {"layout": world.name, "robot": str(world.robot.urdf.name), "margin": MARGIN, "jaw": cc.jaw,
            "verdict": "pass" if n_ok == len(slots) else "fail", "passed": n_ok, "total": len(slots),
            "slots": slots}

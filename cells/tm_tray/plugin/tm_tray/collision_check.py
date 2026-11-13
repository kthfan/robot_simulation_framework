"""M3: is every slot's pick and place collision-free in the cell? (CPU port)

Port of AI-Robotics-V2 ``program/collision/check_cell.py`` onto the framework:
EAIK branches (``rsf.kinematics``), exact meshes on the CPU
(``rsf.collision.coal_backend``) instead of cuRobo, and contact rules instead of
three checkers and one world per obstacle group. Semantics kept (D28, D29):

* every exact, in-limit IK branch of every phase is a candidate;
* everything keeps ``MARGIN`` except the target jig and target tray (margin 0,
  the plate goes under the tray); the jaws may touch the target tray (they clamp
  its sides); the arm base rests on its mount plate;
* while carried, the target tray rides on the plate, attached to the flange; it
  must keep ``MARGIN`` from everything but its own jig (margin 0);
* a pick passes when ``pre_insert -> insert [-> push] -> lift`` have one
  continuous branch chain, a place when ``place_lift -> place_down -> retreat``
  do, and every other phase has at least one passing branch;
* self-collision: exact link meshes, non-adjacent bodies (cuRobo used spheres).

With the push plate (user 2026-11-12) each phase also says where the target
tray is (:class:`sequence.Phase`): on its jig, shifted (placed shallow, pushed
deep), or registered against the plate. In ``push`` and ``place_down`` the
plate touches the tray on purpose: contact is allowed, penetration is not.

The jaws are at the open command throughout (``JAW_Q_OPEN``), as in M3.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from rsf.collision import ContactPolicy, ContactRule
from rsf.collision.coal_backend import CoalChecker
from rsf.core import se3
from rsf.planning.cartesian import walk
from rsf.planning.chain import ContinuityRule, best_chain
from rsf.model.urdf import UrdfModel

from . import frames as F
from . import hardware as A
from . import sequence as SEQ
from .geometry import slot_of
from .tool import tool_geometry

MARGIN = 0.005
#: Touching counts as contact, not penetration, up to this (numerical) depth.
CONTACT_TOL = 1e-6
PICK_CHAIN = ("pre_insert", "insert", "push", "lift")
PLACE_CHAIN = ("place_lift", "place_down", "retreat")
CARRIED_PHASES = ("lift", "withdraw", "carry")    # D28 grasp, kept for tools that use it
JAWS = ("right_gripper_joint", "left_gripper_joint")
JAW_LINKS = ("right_gripper", "left_gripper")
CONTINUITY = ContinuityRule(sign_joints=(2, 4), max_step=1.0)
#: Largest joint change between two samples of a checked motion (rad).
PATH_MAX_STEP = 0.2


def carried_tray_pose(centre_z: float) -> np.ndarray:
    """Tray frame (rails' underside centre; x across, y along, z up) in the flange
    frame while it rides on the plate: recess on the plate's upper face, tray
    centre at flange z *centre_z* (cell_core.scene.carried_tray_box)."""
    R = F.USD_FLANGE_R0.T        # tray x -> flange x, tray y -> flange -z, tray z -> flange y
    return F.make_pose(R, (0.0, A.PLATE_FY_TOP - A.TRAY_RECESS, centre_z))


def policy(key: str, *, contact: bool = False) -> ContactPolicy:
    rules = [
        ContactRule(allow=True, links=JAW_LINKS, names=(f"tray/{key}/*",), label="jaws clamp the target tray"),
        ContactRule(margin=-CONTACT_TOL if contact else 0.0, names=(f"tray/{key}/*",),
                    label="target tray (pushed)" if contact else "target tray"),
        ContactRule(margin=0.0, names=(f"jig/{key}/*",), label="target jig"),
        ContactRule(allow=True, links=("link_0",), tags=("robot_seat",), label="base on its mount"),
    ]
    return ContactPolicy(MARGIN, rules)


def path_policy(clearance: float) -> ContactPolicy:
    """Along a motion with its own clearance (the tilt, user 2026-11-13): every
    obstacle keeps *clearance*, the target jig too; only the base sits on its mount."""
    return ContactPolicy(clearance, [ContactRule(allow=True, links=("link_0",), tags=("robot_seat",),
                                                 label="base on its mount")])


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

    def judge(self, key: str, phase, q6) -> dict:
        self.checker.policy = policy(key, contact=phase.contact)
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

    def _setup(self, key: str, phase) -> None:
        """The world as *phase* has it: the target tray seated (shifted) or carried."""
        tray = self.world.object_at(key)
        if tray is None:
            self.checker.set_obstacles(self.world.solids())
            self.checker.detach("tray")
            return
        others = self.world.solids(exclude_objects=[tray.name])
        if phase.tray == "carried":
            self.checker.set_obstacles(others)
            self.checker.attach("tray", "flange", tray.solids, carried_tray_pose(phase.carried_z))
        else:
            moved = replace(tray, T=tray.T @ se3.make_T(None, (0.0, phase.shift, 0.0)))  # jig +y: toward the robot
            self.checker.set_obstacles(others + moved.world_solids())
            self.checker.detach("tray")

    def _run(self, key: str, phases: list) -> dict:
        out = {}
        for i, ph in enumerate(phases):
            self._setup(key, ph)
            if ph.path and i > 0:
                out[ph.name] = self._along(key, ph, out[phases[i - 1].name])
            else:
                out[ph.name] = [self.judge(key, ph, q) for q in self.arm.ik(ph.T, expand_turns=False)]
        self.checker.detach("tray")
        return out

    def _along(self, key: str, ph, previous: list) -> list:
        """*ph* reached by following ``ph.path`` from every passing branch of the
        previous phase: one continuous IK walk, every sample collision-checked."""
        along = (policy(key, contact=ph.contact) if ph.path_clearance is None
                 else path_policy(ph.path_clearance))
        out = []
        for i, b in enumerate(previous):
            if not b["ok"]:
                continue
            self.checker.policy = along          # judge() below switches it; set it per branch
            self.checker._rule_cache.clear()
            qs = walk(self.arm, ph.path, b["q"], max_step=PATH_MAX_STEP, rule=CONTINUITY)
            if qs is None:
                out.append({"q": None, "ok": False, "score": -1.0, "from": i, "path": "no continuous IK",
                            "worst": None, "violations": [], "self": [], "groups": {}})
                continue
            worst, bad = None, None
            for k, q in enumerate(qs):
                r = self.checker.check(self._q(q))
                if r.worst is not None and (worst is None or r.worst.slack < worst.slack):
                    worst = r.worst
                if not r.ok and bad is None:
                    bad = (k, r.violations[:1] + r.self_violations[:1])
            rec = self.judge(key, ph, qs[-1])
            rec.update({"from": i, "path_min_slack": None if worst is None else round(worst.slack, 5),
                        "path_worst": None if worst is None else (worst.link, worst.obstacle, round(worst.distance, 5))})
            if bad is not None:
                v = bad[1][0]
                rec.update({"ok": False, "path": f"sample {bad[0]}/{len(qs) - 1}: {v.link} vs {v.obstacle} "
                                                 f"{v.distance * 1e3:.1f} mm (needs {v.margin * 1e3:.1f})"})
            out.append(rec)
        return out

    def _chain(self, results: dict, names) -> tuple:
        names = [n for n in names if n in results]
        idx, qs = best_chain([results[n] for n in names], self.arm.lower, self.arm.upper, CONTINUITY)
        return names, idx, qs

    def check_slot(self, feature) -> dict:
        key = feature.key
        slot = slot_of(feature)
        pick = self._run(key, SEQ.pick_phases(self.world.process, slot, self.tool))
        place = self._run(key, SEQ.place_phases(self.world.process, slot, self.tool))
        missing = ([f"pick/{p}" for p, bs in pick.items() if not any(b["ok"] for b in bs)]
                   + [f"place/{p}" for p, bs in place.items() if not any(b["ok"] for b in bs)])
        pn, pidx, pqs = self._chain(pick, PICK_CHAIN)
        ln, lidx, lqs = self._chain(place, PLACE_CHAIN)
        rnd = lambda qs: None if qs is None else [[round(float(v), 6) for v in q] for q in qs]  # noqa: E731
        return {"slot": key, "ok": pidx is not None and lidx is not None and not missing, "missing": missing,
                "chain": None if pidx is None else list(pidx), "chain_phases": pn, "chain_q": rnd(pqs),
                "place_chain": None if lidx is None else list(lidx), "place_chain_phases": ln,
                "place_chain_q": rnd(lqs),
                "phases": {"pick": pick, "place": place}}


def check_cell(world, arm=None, select=None) -> dict:
    from rsf.kinematics import load_arm
    arm = arm or load_arm(world)
    cc = CellChecker(world, arm)
    slots = [cc.check_slot(f) for f in world.features(select or {"kind": "slot"})]
    n_ok = sum(s["ok"] for s in slots)
    return {"layout": world.name, "robot": str(world.robot.urdf.name), "margin": MARGIN, "jaw": cc.jaw,
            "process": dict(world.process), "tool": cc.tool,
            "verdict": "pass" if n_ok == len(slots) else "fail", "passed": n_ok, "total": len(slots),
            "slots": slots}

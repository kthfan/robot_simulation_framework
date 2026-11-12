"""Write the configurations the cuRobo cross-check compares, with the CPU answer.

For a sample of slots, every IK branch of every grasp phase of the TM tray
cell (strict, D29 gripper). The CPU reference is the signed distance from the
robot's meshes (all links but the base, which rests on its mount) to the whole
scene of that phase, capped at 0.1 m: the quantity cuRobo's
``get_mesh_collision_distance`` reports. No contact rules apply here; this
compares geometry, not verdicts.

    PYTHONPATH=src python tools/kaggle/make_crosscheck_cases.py --out tools/kaggle/crosscheck_cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsf.collision import ContactPolicy, ContactRule  # noqa: E402
from rsf.collision.coal_backend import CoalChecker  # noqa: E402
from rsf.kinematics import load_arm  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402

LAYOUT = "cells/tm_tray/layouts/strict.yaml"
ROBOT = "cells/tm_tray/assets/tm5s_egu50_no_air_cylinder.urdf"
SLOTS = ["incoming/L0/C0", "incoming/L2/C3", "outgoing/L0/C5", "outgoing/L3/C0", "outgoing/L3/C5",
         "outgoing/L1/C2", "table/L0/C0"]
CAP = 0.1
JAW = 0.057675


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    world = load_world(ROOT / LAYOUT, {"robot.urdf": str((ROOT / ROBOT).resolve())})
    arm = load_arm(world)
    import rsf_plugins.tm_tray.sequence as SEQ
    from rsf_plugins.tm_tray.collision_check import CARRIED_PHASES
    from rsf_plugins.tm_tray.geometry import slot_of
    ck = CoalChecker(world.robot.urdf, world.robot.base,
                     ContactPolicy(0.0, [ContactRule(allow=True, links=("link_0",))]), cutoff=CAP)
    cases = []
    for key in SLOTS:
        f = world.feature(key)
        poses = SEQ.grasp_sequence(world.process, slot_of(f))
        for carried in (False, True):
            tray = world.object_at(key)
            ck.set_obstacles(world.solids(exclude_objects=[tray.name] if carried else []))
            for phase in SEQ.GRASP_PHASES:
                if (phase in CARRIED_PHASES) != carried:
                    continue
                for q in arm.ik(poses[phase], expand_turns=False):
                    qd = dict(zip(arm.joint_names, q))
                    qd.update({"right_gripper_joint": JAW, "left_gripper_joint": JAW})
                    r = ck.check(qd, self_collision=False)
                    d = min(CAP, r.worst.distance) if r.worst else CAP
                    cases.append({"slot": key, "phase": phase, "carried": carried,
                                  "q": [float(v) for v in q], "cpu_distance": d,
                                  "cpu_link": r.worst.link if r.worst and d < CAP else None,
                                  "cpu_obstacle": r.worst.obstacle if r.worst and d < CAP else None})
    Path(a.out).write_text(json.dumps({"layout": LAYOUT, "robot": ROBOT, "jaw": JAW, "cap": CAP,
                                       "joint_names": arm.joint_names, "cases": cases}, indent=0))
    print(f"{len(cases)} cases -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

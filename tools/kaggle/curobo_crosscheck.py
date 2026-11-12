"""GPU side of the CPU/cuRobo collision cross-check (runs on Kaggle).

Reads ``crosscheck_cases.json`` (configurations plus the CPU's signed distance
from the robot meshes to the scene), builds a cuRobo robot model of the same
URDF with ``RobotBuilder``, loads each case's scene from the same World
(``rsf.compile.curobo``), and asks cuRobo's exact mesh check
(``get_mesh_collision_distance``) for the same distances.

Agreement criteria (cuRobo dev/main semantics, docs/guides/mesh_collision.md):
* CPU distance > 1 mm: cuRobo within 0.5 mm of it (exact for cuboids; float32);
* CPU distance <= 0 (penetration): cuRobo <= 0.5 mm (cuRobo reports the deepest
  vertex inside, or 0 when only edges cross).

    python tools/kaggle/curobo_crosscheck.py --out results/crosscheck.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TOL_POS = 5e-4


def build_robot(urdf: Path, jaw: float) -> dict:
    import yaml
    from curobo.robot_builder import RobotBuilder
    t = time.time()
    b = RobotBuilder(str(urdf), str(urdf.parent), tool_frames=["flange"])
    b.fit_collision_spheres(sphere_density=0.5, use_collision_mesh=True, iterations=50)
    b.compute_collision_matrix(num_samples=200)
    cfg = b.build()
    out = Path("/tmp/rsf_tm5s_egu50.yml")
    b.save(cfg, str(out))
    robot = yaml.safe_load(out.read_text())
    k = robot["kinematics"]
    k["lock_joints"] = {"right_gripper_joint": jaw, "left_gripper_joint": jaw}
    k["mesh_collision_ignored_link_names"] = ["link_0"]
    k["max_triangles_per_link"] = 20000          # no decimation: compare exact meshes
    print(f"robot model built in {time.time() - t:.1f}s; collision links {k.get('collision_link_names')}", flush=True)
    return robot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "tools" / "kaggle" / "crosscheck_cases.json"))
    ap.add_argument("--out", default="results/crosscheck.json")
    a = ap.parse_args(argv)
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.scene import Scene
    from curobo.types import JointState
    from rsf.compile.curobo import world_scene
    from rsf.model.layout import load_world

    data = json.loads(Path(a.cases).read_text())
    world = load_world(ROOT / data["layout"], {"robot.urdf": str((ROOT / data["robot"]).resolve())})
    robot = build_robot((ROOT / data["robot"]).resolve(), data["jaw"])
    ik = InverseKinematics(InverseKinematicsCfg.create(
        robot=robot, scene_model=world_scene(world), mesh_collision_check=True,
        collision_cache={"cuboid": 800}, num_seeds=4, use_cuda_graph=False))
    names = list(ik.joint_names)
    order = [data["joint_names"].index(n) for n in names]
    print("cuRobo joints", names, flush=True)

    groups = defaultdict(list)
    for i, c in enumerate(data["cases"]):
        groups[(c["slot"], c["carried"])].append(i)
    gpu = [None] * len(data["cases"])
    gpu_link = [None] * len(data["cases"])
    for (slot, carried), idx in groups.items():
        tray = world.object_at(slot)
        ik.update_world(Scene.create(world_scene(world, exclude_objects=[tray.name] if carried else [])))
        q = torch.tensor([[data["cases"][i]["q"][j] for j in order] for i in idx], device="cuda", dtype=torch.float32)
        d, links = ik.get_mesh_collision_distance(JointState.from_position(q, joint_names=names),
                                                  max_distance=data["cap"])
        for k, i in enumerate(idx):
            gpu[i] = float(d.reshape(-1)[k])
            gpu_link[i] = links[k] if isinstance(links, list) else None

    rows, bad = [], []
    for c, g, gl in zip(data["cases"], gpu, gpu_link):
        cpu = c["cpu_distance"]
        if cpu > 1e-3:
            ok = abs(g - cpu) <= TOL_POS or (cpu >= data["cap"] - 1e-9 and g >= data["cap"] - TOL_POS)
        elif cpu <= 0.0:
            ok = g <= TOL_POS
        else:
            ok = g <= cpu + TOL_POS
        rows.append({**c, "gpu_distance": g, "gpu_link": gl, "agree": ok})
        if not ok:
            bad.append(rows[-1])
    pos = [r for r in rows if 1e-3 < r["cpu_distance"] < data["cap"] - 1e-9]
    summary = {"cases": len(rows), "agree": sum(r["agree"] for r in rows),
               "positive_cases": len(pos),
               "max_abs_diff_positive": max((abs(r["gpu_distance"] - r["cpu_distance"]) for r in pos), default=None),
               "penetrating_cases": sum(r["cpu_distance"] <= 0 for r in rows)}
    print("SUMMARY", json.dumps(summary), flush=True)
    for r in bad[:15]:
        print("DISAGREE", r["slot"], r["phase"], f"cpu={r['cpu_distance']:.5f} ({r['cpu_link']} vs {r['cpu_obstacle']})",
              f"gpu={r['gpu_distance']:.5f} ({r['gpu_link']})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=0))
    return 0 if not bad else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)

"""Plan the free segments of an rsf job with cuRobo (runs on a GPU, e.g. Kaggle).

Input: a job written by ``rsf.planning.plan.write_job`` (``rsf.curobo_job/1``):
the robot URDF, its base, the operating-limit margin, and per segment the start
and goal joints, the locked joints (jaws), the scene (cuRobo cuboids, robot base
frame, stored once in ``scenes``) and the carried object (boxes in a link frame) if any.

For every distinct (locked joints, carried object) the script writes a URDF
variant -- joint limits moved inside by the margin, the carried object as a
fixed link of boxes -- builds a cuRobo model of it with ``RobotBuilder`` and a
``MotionPlanner`` with the exact mesh checks on (scene, link self-collision, meshes
decide success: ``make_planner``); then ``plan_cspace`` per
segment. Output ``{name: {success, q, dt, seconds, attempts}}``. The CPU checks
every returned path again before it is used (``rsf.planning.plan.assemble``).

    python tools/kaggle/curobo_free_plan.py --job tools/kaggle/jobs/x.json --out results/x_paths.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def variant_urdf(job: dict, lock: dict, carried, workdir: Path) -> Path:
    from rsf.model.primitives import write_box_obj
    from rsf.model.urdf import Geometry, Joint, Link, UrdfModel
    urdf = Path(job["robot_urdf"])
    if not urdf.is_absolute():                               # relative to the repository
        urdf = ROOT / urdf
    m = UrdfModel.from_file(urdf)
    margin = math.radians(job.get("limit_margin_deg", 0.0))
    for j in m.joints.values():
        if j.type == "revolute" and j.name not in lock:
            j.lower, j.upper = j.lower + margin, j.upper - margin
    if carried:
        geos = []
        for i, b in enumerate(carried["boxes"]):
            d = np.array(b["dims"]) / 2.0
            p = write_box_obj(workdir / f"carried_{i}.obj", -d, d)
            geos.append(Geometry("mesh", np.array(b["pose_in_link"]), name=f"carried_{i}", filename=str(p), path=p))
        m.add_link(Link("carried_object", visuals=list(geos), collisions=list(geos), mass=1.0))
        m.add_joint(Joint("carried_object_joint", "fixed", carried["link"], "carried_object", np.eye(4)))
    tag = hashlib.sha1(json.dumps([lock, carried], sort_keys=True).encode()).hexdigest()[:10]
    out = urdf.parent / f"_rsf_variant_{tag}.urdf"          # beside the original: relative mesh paths hold
    out.write_text(m.to_xml(relative_to=out.parent))
    return out


#: exact-mesh options (set from the command line in ``main``)
MESH = {"self": True, "only_success": True, "self_margin": 0.0}


def make_planner(robot: dict, scene: dict, job: dict):
    """A MotionPlanner with cuRobo's exact mesh checks: robot meshes vs the scene, link
    meshes vs each other, and (``mesh_only_success``) the meshes, not the spheres, decide
    whether a solution succeeds; the spheres still steer the optimisers."""
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    return MotionPlanner(MotionPlannerCfg.create(
        robot=robot, scene_model=scene, collision_cache={"cuboid": 1000},
        mesh_collision_check=True, mesh_collision_margin=float(job["mesh_collision_margin"]),
        mesh_self_collision_check=MESH["self"], mesh_self_collision_margin=MESH["self_margin"],
        mesh_only_success=MESH["only_success"], use_cuda_graph=False))


#: links the carried object touches by design (held by the jaws, against the push plate)
HOLDING_LINKS = ["link_6", "gripper_body", "left_gripper", "right_gripper"]


def build_robot(urdf: Path, tool_frame: str, lock: dict, carried: bool = False) -> dict:
    import yaml
    from curobo.robot_builder import RobotBuilder
    t = time.time()
    b = RobotBuilder(str(urdf), str(urdf.parent), tool_frames=[tool_frame])
    b.fit_collision_spheres(sphere_density=0.5, use_collision_mesh=True, iterations=50)
    hold = {"carried_object": list(HOLDING_LINKS)} if carried else None
    b.compute_collision_matrix(num_samples=200, custom_ignore=hold)
    if MESH["self"]:                                         # exact link-mesh self-collision pairs
        ign = b.compute_mesh_collision_matrix(num_samples=1000, custom_ignore=hold)
        print(f"  mesh self-collision ignore: {ign}", flush=True)
    cfg = b.build()
    out = Path("/tmp") / (urdf.stem + ".yml")
    b.save(cfg, str(out))
    robot = yaml.safe_load(out.read_text())
    k = robot["kinematics"]
    k["lock_joints"] = dict(lock)
    k["mesh_collision_ignored_link_names"] = ["link_0"]       # bolted to its mount plate
    k["max_triangles_per_link"] = 20000
    print(f"  robot model {urdf.name}: {time.time() - t:.1f}s, links {k.get('collision_link_names')}", flush=True)
    return robot


def job_margin(path) -> float:
    return float(json.loads(Path(path).read_text())["mesh_collision_margin"])


def main_diag(job: dict, segs: list, work: Path, attempts: int) -> dict:
    """``{name: success}`` for *segs* planned with the jaws locked as given but no carried object."""
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene
    from curobo.types import JointState
    lock = segs[0]["lock_joints"]
    robot = build_robot(variant_urdf(job, lock, None, work), job["tool_frame"], lock)
    mp = make_planner(robot, segs[0]["scene"], job)
    names = list(mp.joint_names)
    dev = dict(device=torch.device("cuda", 0), dtype=torch.float32)
    out = {}
    for s in segs:
        try:
            mp.update_world(Scene.create(s["scene"]))
            q0 = torch.tensor([s["q_start"]], **dev)
            q1 = torch.tensor([s["q_goal"]], **dev)
            res = mp.plan_cspace(JointState.from_position(q1, joint_names=names),
                                 JointState.from_position(q0, joint_names=names), max_attempts=attempts)
            out[s["name"]] = {"success": bool(res is not None and res.success.reshape(-1)[0]),
                              "info": str(getattr(res, "debug_info", "") or "")}
        except Exception as e:  # noqa: BLE001
            out[s["name"]] = {"success": False, "info": f"{type(e).__name__}: {e}"}
        print(f"  without tray {'ok  ' if out[s['name']]['success'] else 'FAIL'} {s['name']} "
              f"{out[s['name']]['info']}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", default="results/paths.json")
    ap.add_argument("--attempts", type=int, default=20)
    ap.add_argument("--no-diagnose", action="store_true")
    ap.add_argument("--no-mesh-self", action="store_true", help="no exact link-mesh self-collision check")
    ap.add_argument("--sphere-success", action="store_true",
                    help="spheres decide success (cuRobo default), not the exact meshes")
    ap.add_argument("--mesh-self-margin", type=float, default=0.0)
    a = ap.parse_args(argv)
    MESH.update({"self": not a.no_mesh_self, "only_success": not a.sphere_success,
                 "self_margin": a.mesh_self_margin})
    print(f"mesh checks: scene margin {job_margin(a.job)} m, self {MESH['self']} "
          f"(margin {MESH['self_margin']} m), mesh decides success {MESH['only_success']}", flush=True)
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene
    from curobo.types import JointState

    job = json.loads(Path(a.job).read_text())
    scenes = job.get("scenes", {})
    for s in job["segments"]:                                  # scenes stored once, by id
        if isinstance(s["scene"], str):
            s["scene"] = scenes[s["scene"]]
    work = Path("/tmp/rsf_variants")
    work.mkdir(parents=True, exist_ok=True)
    groups = {}
    for s in job["segments"]:
        key = json.dumps([s["lock_joints"], s["carried"]], sort_keys=True)
        groups.setdefault(key, []).append(s)
    results = {}
    for key, segs in groups.items():
        lock, carried = json.loads(key)
        print(f"variant: lock {lock}, carried {'yes' if carried else 'no'}: {len(segs)} segments", flush=True)
        robot = build_robot(variant_urdf(job, lock, carried, work), job["tool_frame"], lock, bool(carried))
        mp = make_planner(robot, segs[0]["scene"], job)
        names = list(mp.joint_names)
        dev = dict(device=torch.device("cuda", 0), dtype=torch.float32)
        for s in segs:
            t = time.time()
            rec = {"success": False}
            try:
                mp.update_world(Scene.create(s["scene"]))
                order = list(range(len(names)))         # job joints are the arm joints in chain order
                q0 = torch.tensor([[s["q_start"][i] for i in order]], **dev)
                q1 = torch.tensor([[s["q_goal"][i] for i in order]], **dev)
                if float((q1 - q0).abs().max()) < 1e-6:
                    rec = {"success": True, "q": [s["q_start"], s["q_goal"]], "dt": 0.0}
                else:
                    res = mp.plan_cspace(JointState.from_position(q1, joint_names=names),
                                         JointState.from_position(q0, joint_names=names), max_attempts=a.attempts)
                    ok = res is not None and bool(res.success.reshape(-1)[0])
                    if ok:
                        ip = res.interpolated_trajectory
                        last = int(res.interpolated_last_tstep.reshape(-1)[0])
                        # the trajectory may carry the locked joints too: pick the arm by name
                        jn = list(getattr(ip, "joint_names", None) or [])
                        P = ip.position.detach().cpu().numpy().astype(float)
                        P = P.reshape(-1, P.shape[-1])[: last + 1]
                        cols = [jn.index(n) for n in names] if jn and all(n in jn for n in names) \
                            else list(range(len(names)))
                        Q = P[:, cols]
                        dt = ip.dt.reshape(-1)[0] if hasattr(ip.dt, "reshape") else ip.dt
                        rec = {"success": True, "q": Q.tolist(), "dt": float(dt)}
                    else:
                        rec["error"] = str(getattr(res, "debug_info", "") or "not solved")
                rec["attempts"] = int(getattr(mp, "_last_planning_attempts", 0) or 0)
            except Exception as e:  # noqa: BLE001
                rec = {"success": False, "error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            rec["seconds"] = round(time.time() - t, 2)
            results[s["name"]] = rec
            print(f"  {'ok  ' if rec['success'] else 'FAIL'} {s['name']} ({rec['seconds']} s)"
                  f"{'' if rec['success'] else ' ' + rec.get('error', '')}", flush=True)
    bad = [s for s in job["segments"] if s["carried"] and not results[s["name"]]["success"]]
    if bad and not a.no_diagnose:                     # is it the tray? the same moves without it
        print(f"diagnose: {len(bad)} carried moves failed; again without the tray", flush=True)
        diag = main_diag(job, bad, work, a.attempts)
        for n, r in diag.items():
            results[n]["without_tray"] = r
    n_ok = sum(r["success"] for r in results.values())
    print(f"SUMMARY {n_ok}/{len(results)} free segments planned", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""P6 prep: the AprilGrid reads in simulation. For every slot, the arm at its
taught far and near read poses (``sight_far``, ``sight_near``), the locating
camera's image rendered in MuJoCo with every label printed on its jig, the
user's detector (kthfan/AprilgridDetector) run on it, and the board pose it
finds compared with the truth.

    MUJOCO_GL=osmesa PYTHONPATH=src python tools/sim_locate.py cells/tm_tray/layouts/strict.yaml \\
        --teach cells/tm_tray/teach/strict.json --out reports/locate_strict.json [--images reports/locate_img]

Needs ``aprilgrid_detector`` (build: maturin, see docs/p6_locating.md) and cv2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

READS = {"sight_far": "standoff_far", "sight_near": "standoff_near"}


def build(world, labels, cams):
    """The MuJoCo cell (static, no physics needed) with the labels and one camera per focus config."""
    import mujoco
    from dataclasses import replace
    from rsf.compile.mjcf import ROBOT_PREFIX, world_spec
    from rsf.sim import vision as V
    from rsf_plugins.tm_tray.vision import sheet_image
    spec = world_spec(world, actuators=False)
    spec.visual.global_.offwidth = max(c["width"] for c in cams.values())
    spec.visual.global_.offheight = max(c["height"] for c in cams.values())
    spec.visual.headlight.ambient = [0.55, 0.55, 0.55]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    for lab in labels:
        V.add_marker(spec, f"label/{lab['key']}", lab["T_sheet"], lab["size"], sheet_image(lab["grid_id"]))
    for name, c in cams.items():
        V.add_camera(spec, ROBOT_PREFIX + c["link"], name, c["T_link_cv"], c["fx"], c["fy"], c["cx"], c["cy"],
                     c["width"], c["height"])
    return spec.compile()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("--teach", required=True, help="slot teach table (rsf teach --skill tm_tray.pick ...)")
    ap.add_argument("--slots", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", help="save the rendered images here")
    a = ap.parse_args(argv)
    import mujoco
    import aprilgrid_detector as agd
    from rsf.compile.mjcf import ROBOT_PREFIX
    from rsf.model import registry
    from rsf.model.layout import load_world
    from rsf.sim import vision as V
    from rsf.sim.judge import pose_error
    world = load_world(a.layout)
    labels = registry.get("vision", "tm_tray.labels")(world)
    cams = {}
    for read, key in READS.items():
        c = registry.get("vision", "tm_tray.camera")(world, float(world.process[key]))
        cams[f"eih_c{c['config']}"] = c
    m = build(world, labels, cams)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, max(c["height"] for c in cams.values()), max(c["width"] for c in cams.values()))
    det = agd.Detector()
    layout = agd.BoardLayout(tags_x=4, tags_y=1, tag_size=0.020, tag_separation=0.0075)
    teach = {s["sequence"]: {p["name"]: p for p in s["points"]} for s in json.loads(Path(a.teach).read_text())["sequences"]}
    adr = {n: m.joint(ROBOT_PREFIX + n).qposadr[0] for n in [f"joint_{i}" for i in range(1, 7)]}
    jaws = [n for n in ("left_gripper_joint", "right_gripper_joint")]
    rows, t0 = [], time.time()
    for lab in labels:
        if a.slots and lab["key"] not in a.slots:
            continue
        pts = teach.get(f"pick:{lab['key']}", {})
        for read, key in READS.items():
            p = pts.get(read)
            if p is None:
                continue
            c = registry.get("vision", "tm_tray.camera")(world, float(world.process[key]))
            name = f"eih_c{c['config']}"
            d.qpos[:] = 0.0
            for i, v in enumerate(p["q"]):
                d.qpos[adr[f"joint_{i + 1}"]] = v
            for n in jaws:
                d.qpos[m.joint(ROBOT_PREFIX + n).qposadr[0]] = p["joints"].get(n, 0.05)
            mujoco.mj_kinematics(m, d)                  # poses only: no contacts needed to render
            mujoco.mj_camlight(m, d)
            img = V.render_grey(m, d, name, renderer=r)
            cam_T = np.eye(4)
            cid = m.camera(name).id
            cam_T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ V.CV_TO_MJ
            cam_T[:3, 3] = d.cam_xpos[cid]
            camera = agd.Camera(fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"])
            boards = det.detect_boards(img, layout, camera)
            mine = [b for b in boards if b.board_id == lab["grid_id"]]
            rec = {"slot": lab["key"], "read": read, "config": c["config"], "grid_id": lab["grid_id"],
                   "boards_seen": sorted(b.board_id for b in boards), "found": bool(mine)}
            if mine:
                b = mine[0]
                Tbc = np.eye(4)
                Tbc[:3, :3] = _rodrigues(np.asarray(b.pose.rvec, float))
                Tbc[:3, 3] = np.asarray(b.pose.translation, float)
                Tw = cam_T @ Tbc
                mm, deg = pose_error(Tw, lab["board_T"])
                rec.update({"pos_err_mm": round(mm, 3), "rot_err_deg": round(deg, 3),
                            "distance_m": round(float(np.linalg.norm(Tbc[:3, 3])), 4),
                            "rmse_px": round(float(b.pose.reprojection_rmse), 3), "coverage": round(float(b.coverage), 3)})
            rows.append(rec)
            if a.images:
                from PIL import Image
                Path(a.images).mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(Path(a.images) / f"{lab['key'].replace('/', '_')}_{read}.png")
            print(f"{lab['key']:16s} {read:10s} c{c['config']}  "
                  + (f"pos {rec['pos_err_mm']:6.2f} mm  rot {rec['rot_err_deg']:5.2f} deg  rmse {rec['rmse_px']:.2f} px"
                     if mine else f"NOT FOUND (saw {rec['boards_seen']})"), flush=True)
    found = [x for x in rows if x["found"]]
    summary = {"reads": len(rows), "found": len(found),
               "pos_err_mm": {"mean": float(np.mean([x["pos_err_mm"] for x in found])) if found else None,
                              "max": float(max((x["pos_err_mm"] for x in found), default=float("nan")))},
               "rot_err_deg": {"mean": float(np.mean([x["rot_err_deg"] for x in found])) if found else None,
                               "max": float(max((x["rot_err_deg"] for x in found), default=float("nan")))},
               "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"layout": a.layout, "teach": a.teach, "summary": summary, "reads": rows}, indent=1))
    print("LOCATE", json.dumps(summary))
    r.close()
    return 0 if len(found) == len(rows) else 1


def _rodrigues(rvec) -> np.ndarray:
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


if __name__ == "__main__":
    sys.exit(main())

"""P6: locate the racks from AprilGrid reads and write the corrected layout.

The cell as it really stands (``--truth``, e.g. a rack moved a few millimetres)
is rendered; the robot only knows the nominal layout and its teach points. For
every slot of the racks to locate:

1. **far read** at the taught ``sight_far`` pose: the board the detector finds,
   against where the nominal layout puts it, gives the slot's correction;
2. **near read** at ``sight_near`` moved by that correction (IK nearest the taught
   joints): the refined correction.

Each rack is rigid, so its displacement (yaw about the vertical, x, y, z: the
racks stand on the floor) is fitted to the near reads of all its slots, and the
corrected layout (``--out-layout``: the nominal layout with ``displace`` on each
located rack) is what the robot is then taught and planned in. With ``--truth``
built from a ``displace`` too, the report states how far each slot's corrected
pose is from the truth.

    MUJOCO_GL=osmesa PYTHONPATH=src python tools/locate_slots.py \\
        --truth cells/tm_tray/layouts/perturbed/strict_racks_moved.yaml \\
        --nominal cells/tm_tray/layouts/strict.yaml --teach cells/tm_tray/teach/strict.json \\
        --racks incoming outgoing --out-layout reports/locate/strict_located.yaml --out reports/locate/strict_located.json
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
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

#: board-frame points used to fit a rack's displacement (m): origin, along the row, down the column
_BOARD_PTS = np.array([[0.0, 0.0, 0.0, 1.0], [0.1125, 0.0, 0.0, 1.0], [0.0, 0.030, 0.0, 1.0], [0.1125, 0.030, 0.0, 1.0]]).T

from rsf.sim.reads import describe, fit_planar, planar  # noqa: E402,F401  (one copy: rsf.sim.reads)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="the cell as it stands (rendered)")
    ap.add_argument("--nominal", required=True, help="the layout the robot was taught in")
    ap.add_argument("--teach", required=True, help="nominal slot teach table (rsf teach --skill tm_tray.pick ...)")
    ap.add_argument("--racks", nargs="+", required=True, help="entities to locate")
    ap.add_argument("--out-layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", help="save the rendered images here")
    a = ap.parse_args(argv)
    import mujoco
    import aprilgrid_detector as agd
    from sim_locate import _rodrigues, build
    from rsf.compile.mjcf import ROBOT_PREFIX
    from rsf.kinematics import load_arm
    from rsf.model import registry
    from rsf.model.layout import load_world
    from rsf.sim import vision as V
    from rsf.sim.judge import pose_error

    truth, nom = load_world(a.truth), load_world(a.nominal)
    labels_t = {l["key"]: l for l in registry.get("vision", "tm_tray.labels")(truth)}
    labels_n = {l["key"]: l for l in registry.get("vision", "tm_tray.labels")(nom)}
    reads = {"sight_far": "standoff_far", "sight_near": "standoff_near"}
    cams = {}
    for read, key in reads.items():
        c = registry.get("vision", "tm_tray.camera")(nom, float(nom.process[key]))
        cams[read] = (f"eih_c{c['config']}", c)
    m = build(truth, list(labels_t.values()), {n: c for n, c in cams.values()})
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, max(c["height"] for _, c in cams.values()), max(c["width"] for _, c in cams.values()))
    det = agd.Detector()
    layout = agd.BoardLayout(tags_x=4, tags_y=1, tag_size=0.020, tag_separation=0.0075)
    arm = load_arm(nom)
    teach = {s["sequence"]: {p["name"]: p for p in s["points"]} for s in json.loads(Path(a.teach).read_text())["sequences"]}
    adr = [m.joint(ROBOT_PREFIX + f"joint_{i}").qposadr[0] for i in range(1, 7)]

    def read_board(q, jaws, read, key, grid_id):
        name, c = cams[read]
        d.qpos[:] = 0.0
        d.qpos[adr] = q
        for n, v in jaws.items():
            d.qpos[m.joint(ROBOT_PREFIX + n).qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        img = V.render_grey(m, d, name, renderer=r)
        if a.images:
            from PIL import Image
            Path(a.images).mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(Path(a.images) / f"{key.replace('/', '_')}_{read}.png")
        cid = m.camera(name).id
        cam_T = np.eye(4)
        cam_T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ V.CV_TO_MJ
        cam_T[:3, 3] = d.cam_xpos[cid]
        boards = det.detect_boards(img, layout, agd.Camera(fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"]))
        mine = [b for b in boards if b.board_id == grid_id]
        if not mine:
            return None, sorted(b.board_id for b in boards)
        Tbc = np.eye(4)
        Tbc[:3, :3] = _rodrigues(np.asarray(mine[0].pose.rvec, float))
        Tbc[:3, 3] = np.asarray(mine[0].pose.translation, float)
        return cam_T @ Tbc, None

    rows, fits, t0 = [], {}, time.time()
    for rack in a.racks:
        P, Q = [], []
        for f in nom.features({"kind": "slot"}):
            if f.entity != rack:
                continue
            key = f.key
            pts = teach.get(f"pick:{key}", {})
            if "sight_far" not in pts or "sight_near" not in pts:
                rows.append({"slot": key, "error": "no taught sight poses"})
                continue
            B_nom, B_true = labels_n[key]["board_T"], labels_t[key]["board_T"]
            gid = labels_n[key]["grid_id"]
            rec = {"slot": key, "rack": rack, "truth": dict(zip(("mm", "deg"), pose_error(B_true, B_nom)))}
            # 1. far read at the taught pose
            far = pts["sight_far"]
            B_far, seen = read_board(np.array(far["q"]), far.get("joints", {}), "sight_far", key, gid)
            if B_far is None:
                rec["error"] = f"far read: board {gid} not found (saw {seen})"
                rows.append(rec)
                continue
            C_far = planar(B_far, B_nom)
            rec["far_error"] = dict(zip(("mm", "deg"), pose_error(C_far @ B_nom, B_true)))
            # 2. near read at the taught near pose moved by the far correction
            near = pts["sight_near"]
            q_t = np.array(near["q"])
            T = C_far @ arm.fk(q_t)
            sols = [np.asarray(q) for q in arm.ik(T)]
            sols = [q for q in sols if np.all(q >= arm.lower) and np.all(q <= arm.upper)]
            if not sols:
                rec["error"] = "near read: no IK for the corrected pose"
                rows.append(rec)
                continue
            q_n = min(sols, key=lambda q: float(np.abs(q - q_t).max()))
            rec["near_joint_shift_deg"] = round(float(np.degrees(np.abs(q_n - q_t).max())), 3)
            B_near, seen = read_board(q_n, near.get("joints", {}), "sight_near", key, gid)
            if B_near is None:
                rec["error"] = f"near read: board {gid} not found (saw {seen})"
                rows.append(rec)
                continue
            C_near = planar(B_near, B_nom)
            rec["near_error"] = dict(zip(("mm", "deg"), pose_error(C_near @ B_nom, B_true)))
            P.append((B_nom @ _BOARD_PTS)[:3])
            Q.append((B_near @ _BOARD_PTS)[:3])
            rows.append(rec)
            print(f"{key:16s} truth {rec['truth']['mm']:5.2f} mm {rec['truth']['deg']:4.2f} deg   "
                  f"far err {rec['far_error']['mm']:5.2f} mm {rec['far_error']['deg']:4.2f} deg   "
                  f"near err {rec['near_error']['mm']:5.2f} mm {rec['near_error']['deg']:4.2f} deg", flush=True)
        if not P:
            print(f"{rack}: no reads", flush=True)
            continue
        D = fit_planar(np.hstack(P), np.hstack(Q))
        ent_t = truth.entity(rack)
        D_true = np.asarray(ent_t.meta.get("displaced", np.eye(4)), float)
        about = np.mean([f.T[:3, 3] for f in nom.features({"kind": "slot"}) if f.entity == rack], axis=0)
        # every slot of the rack: corrected jig pose versus the truth
        errs = [pose_error(D @ f.T, truth.feature(f.key).T) for f in nom.features({"kind": "slot"}) if f.entity == rack]
        fits[rack] = {"D": D.tolist(), "about_slot_centroid": [round(float(v), 4) for v in about],
                      "estimated": describe(D, about), "truth": describe(D_true, about),
                      "slots": len(errs), "reads": len(P),
                      "jig_error_mm": {"max": round(max(e[0] for e in errs), 3), "mean": round(float(np.mean([e[0] for e in errs])), 3)},
                      "jig_error_deg": {"max": round(max(e[1] for e in errs), 4)}}
        print(f"{rack}: estimated {fits[rack]['estimated']}  truth {fits[rack]['truth']}  "
              f"jig error max {fits[rack]['jig_error_mm']['max']} mm / {fits[rack]['jig_error_deg']['max']} deg", flush=True)
    r.close()

    # the corrected layout: the nominal one with each located rack displaced as estimated
    out = Path(a.out_layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(Path(a.nominal).resolve(), out.parent.resolve())
    lines = ["# Written by tools/locate_slots.py: the nominal layout with the racks where the",
             "# AprilGrid reads put them. Teach and plan in this layout.",
             "schema: rsf.layout/1", f"extends: {rel}", f"name: {nom.name}_located", "entities:"]
    for rack, fit in fits.items():
        mat = [[round(float(v), 9) for v in row] for row in fit["D"]]
        lines.append(f"  {rack}:")
        lines.append(f"    displace: {{matrix: {json.dumps(mat)}}}   # {fit['estimated']}")
    out.write_text("\n".join(lines) + "\n")
    bad = [x for x in rows if "error" in x]
    summary = {"slots": len(rows), "located": len(rows) - len(bad), "racks": fits, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"truth": a.truth, "nominal": a.nominal, "teach": a.teach, "layout": str(out),
                                       "summary": summary, "reads": rows}, indent=1, default=float))
    print("LOCATE", json.dumps({k: v for k, v in summary.items() if k != "racks"}), "->", out)
    return 0 if not bad and fits else 1


if __name__ == "__main__":
    sys.exit(main())

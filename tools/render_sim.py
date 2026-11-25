"""Pictures of a simulated run (MuJoCo, offscreen): stills at the phases of a transfer and a GIF.

    MUJOCO_GL=osmesa PYTHONPATH=src python tools/render_sim.py cells/tm_tray/layouts/strict.yaml \\
        cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_inspect_and_sort_gpu.json \\
        --transfer 1 --out docs/images/p5

Stills: a close-up following the gripper and an overview, at the end of each
phase of the transfer (one sheet per view). GIF: the whole transfer, one frame
every ``--gif-dt`` simulated seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

def colour(m, objects):
    import mujoco

    from rsf.sim.mujoco_cell import style
    style(m, objects)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 960
    return mujoco


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("workflow")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--transfer", type=int, default=1, help="1-based")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--gif-dt", type=float, default=0.4)
    ap.add_argument("--size", type=int, nargs=2, default=(640, 480))
    a = ap.parse_args(argv)
    from PIL import Image, ImageDraw
    from rsf.model.layout import load_world
    from rsf.model.workflow import Workflow
    from rsf.sim import judge as J
    from rsf.sim.mujoco_cell import MujocoCell
    world = load_world(a.layout)
    wf = Workflow.from_file(a.workflow)
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    traj = json.loads(Path(a.plan).read_text())
    tr = J.transfers(world, wf, traj["segments"], occ)[a.transfer - 1]
    phases = [s for s in traj["segments"] if tr.t0 <= s["t0"] and s["t1"] <= tr.t1 and s["kind"] in ("line", "path", "free")]
    shots = {round(s["t1"], 2): s["name"].split(":")[-1] if s["kind"] != "free" else s["name"].split(":", 1)[1]
             for s in phases}
    cell = MujocoCell(world, occ)
    mj = colour(cell.m, set(cell.object_names))
    W, H = a.size
    r = mj.Renderer(cell.m, H, W)
    close, wide = mj.MjvCamera(), mj.MjvCamera()
    wide.lookat[:] = world.robot.base[:3, 3] + np.array([0.0, 0.0, 0.3])
    wide.distance, wide.azimuth, wide.elevation = 2.6, 135.0, -28.0
    close.distance, close.elevation = 0.9, -8.0
    stills = {"close": [], "wide": []}
    gif, next_gif = [], [tr.t0]

    def snap(cam, d):
        r.update_scene(d, camera=cam)
        return r.render().copy()

    base = world.robot.base[:3, 3]
    taken = set()

    def on_sample(t, m, d):
        if t < tr.t0 - 1e-6 or t > tr.t1 + 0.3:
            return
        f = d.xpos[cell.flange_id]
        z = d.xmat[cell.flange_id].reshape(3, 3)[:, 2]          # the plate's long axis (approach)
        close.lookat[:] = f + 0.12 * z
        # from the tray's side, nearly level: a level tray looks level and its tilt reads true
        # (a view along the tray foreshortens it and can make it look upright)
        close.azimuth = float(np.degrees(np.arctan2(z[1], z[0]))) + 90.0
        for ts, label in shots.items():
            if ts not in taken and t >= ts - 1e-9:
                taken.add(ts)
                stills["close"].append((label, snap(close, d)))
                stills["wide"].append((label, snap(wide, d)))
        if t >= next_gif[0]:
            gif.append(Image.fromarray(snap(close, d)).resize((W // 2 * 2 // 2, H // 2)))
            next_gif[0] += a.gif_dt

    cell.run(traj, t_end=tr.t1 + 0.3, on_sample=on_sample)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for view, imgs in stills.items():
        cols = 4
        rows = (len(imgs) + cols - 1) // cols
        tw, th = W // 2, H // 2
        sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), "white")
        dr = ImageDraw.Draw(sheet)
        for k, (label, img) in enumerate(imgs):
            x, y = (k % cols) * tw, (k // cols) * (th + 18)
            sheet.paste(Image.fromarray(img).resize((tw, th)), (x, y + 18))
            dr.text((x + 4, y + 3), label, fill="black")
        p = out.with_name(f"{out.name}_{view}.png")
        sheet.save(p)
        print(f"{view}: {len(imgs)} phases -> {p}")
    if gif:
        p = out.with_name(f"{out.name}.gif")
        gif[0].save(p, save_all=True, append_images=gif[1:], duration=int(1000 * a.gif_dt / 4), loop=0)
        print(f"gif: {len(gif)} frames (x4 speed) -> {p}")
    r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

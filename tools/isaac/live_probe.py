"""Which way of rendering the live views works with this Isaac Sim, and what it costs.

    python tools/isaac/live_probe.py [--modes play,step,update] [--out reports/isaac/live_probe.json]
    (Isaac Sim's standalone Python: ~/isaacsim/python.sh tools/isaac/live_probe.py)

Each mode (``IsaacParams.render_mode``) runs in its own process (an Isaac app per mode):
a 2 s move of joints 1-2 replayed three times on the strict cell, the second time with
the overview view rendered every 0.1 s. Per mode: whether the pictures are blank,
whether they show the arm move, seconds per render, and whether rendering changed
the run (against the spread of the two runs without it). PNGs of the first and last
frame next to the report. The mode to use is the first with ``ok``: pass it as
``--set render_mode='"MODE"'`` to tools/sim_workflow.py (or make it the default).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def one(mode: str, out_dir: Path) -> dict:
    import numpy as np

    from rsf.model.layout import load_world
    from rsf.model.workflow import Workflow
    from rsf.sim.common import overview
    from rsf.sim.isaac_cell import IsaacCell, IsaacParams
    r = {"mode": mode, "ok": False}
    t0 = time.time()
    world = load_world(ROOT / "cells/tm_tray/layouts/strict.yaml")
    wf = Workflow.from_file(ROOT / "cells/tm_tray/workflows/inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    cell = IsaacCell(world, occ, IsaacParams(render_mode=mode))
    r["t_app"] = round(time.time() - t0, 1)
    cell.add_view(overview(world, width=640, height=360))
    names = [f"joint_{i}" for i in range(1, 7)] + ["left_gripper_joint", "right_gripper_joint"]
    q0, q1 = [0.0] * 6 + [0.0577] * 2, [0.6, -0.3] + [0.0] * 4 + [0.0577] * 2
    traj = {"joint_names": names, "t": [0.0, 2.0], "q": [q0, q1], "qd": [[0.0] * 8] * 2}
    w = time.time()
    warm = cell.render("overview")                    # starts the cell, first frame (shaders)
    r["t_first_frame"] = round(time.time() - w, 1)
    r["first_frame"] = None if warm is None else {"shape": list(warm.shape), "max": int(warm.max())}
    a = cell.run(traj)
    frames, times = [], []

    def hook(t, *_):
        if t >= 0 and (not frames or t - frames[-1][0] >= 0.1 - 1e-9):
            s = time.time()
            img = cell.render("overview")
            times.append(time.time() - s)
            frames.append((t, None if img is None else img.copy()))
    b = cell.run(traj, on_sample=hook)
    c = cell.run(traj)
    imgs = [f for _, f in frames if f is not None]
    r["frames"] = len(frames)
    r["none_frames"] = sum(f is None for _, f in frames)
    r["blank_frames"] = sum(int(f[::8, ::8].max()) < 8 for f in imgs)
    r["s_per_render"] = round(float(np.median(times)), 4) if times else None
    spread = float(np.abs(a.q - c.q).max())
    r["q_spread_without_render_rad"] = spread
    r["q_change_with_render_rad"] = float(np.abs(a.q - b.q).max())
    r["run_unchanged"] = r["q_change_with_render_rad"] <= max(1e-9, 2.0 * spread)
    if imgs:
        first, last = imgs[0], imgs[-1]
        r["mean_first"], r["mean_last"] = round(float(first.mean()), 1), round(float(last.mean()), 1)
        r["moved"] = round(float(np.abs(last.astype(int) - first.astype(int)).mean()), 2)
        try:
            from PIL import Image
            for tag, img in (("first", first), ("last", last)):
                p = out_dir / f"live_probe_{mode}_{tag}.png"
                Image.fromarray(img).save(p)
                r[f"png_{tag}"] = str(p)
        except Exception as e:  # noqa: BLE001
            r["png_error"] = repr(e)
    r["ok"] = bool(imgs and r["blank_frames"] == 0 and r.get("moved", 0) > 0.5 and r["run_unchanged"])
    r["t_total"] = round(time.time() - t0, 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="play,step,update")
    ap.add_argument("--out", default="reports/isaac/live_probe.json")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.one:                                         # a child: one mode, one Isaac app
        try:
            r = one(a.one, out.parent)
        except Exception as e:  # noqa: BLE001
            r = {"mode": a.one, "ok": False, "error": repr(e), "trace": traceback.format_exc()}
        print("[live_probe] " + json.dumps(r), flush=True)
        os._exit(0)                                   # closing Kit can hang; the result is out
    results = []
    for mode in a.modes.split(","):
        print(f"== mode {mode}", flush=True)
        p = subprocess.run([sys.executable, __file__, "--one", mode, "--out", str(out)],
                           capture_output=True, text=True, cwd=ROOT)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("[live_probe] ")]
        r = json.loads(lines[-1][len("[live_probe] "):]) if lines else {
            "mode": mode, "ok": False, "error": f"no result (exit {p.returncode})", "stderr_tail": p.stderr[-3000:]}
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("trace", "stderr_tail")}), flush=True)
        if r.get("trace"):
            print(r["trace"], flush=True)
    best = next((r["mode"] for r in results if r.get("ok")), None)
    out.write_text(json.dumps({"use": best, "results": results}, indent=1))
    print(f"[live_probe] use render_mode={best!r} -> {out}", flush=True)
    return 0 if best else 1


if __name__ == "__main__":
    sys.exit(main())

"""Which way of rendering the live views works with this Isaac Sim, and what it costs.

    python tools/isaac/live_probe.py [--modes play,play-nosync,step] [--out reports/isaac/live_probe.json]
    (Isaac Sim's standalone Python: ~/isaacsim/python.sh tools/isaac/live_probe.py)

Each mode (``IsaacParams.render_mode``; ``-nosync``: without ``render_sync``) runs in its
own process (an Isaac app per mode): the first 10 s of the strict plan (the arm goes from
home to the incoming rack) replayed three times, the second time with the overview view
rendered every 0.5 s. Per mode: whether the pictures are blank, how much of the picture
changes (``changed_pct``: pixels differing by more than 10 levels between first and last
frame; ``unchanged``: frames equal to the one before), how far USD (what the renderer
draws) lags PhysX at the end (``usd_lag_m``), seconds per render, and whether rendering
changed the run (more than 1e-6 rad, or the spread of two runs without it). PNGs of the
first and last frame next to the report. The mode to use is the first with ``ok``: pass it as
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
    import gzip

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
    rmode, _, flag = mode.partition("-")
    cell = IsaacCell(world, occ, IsaacParams(render_mode=rmode, render_sync=flag != "nosync"))
    r["t_app"] = round(time.time() - t0, 1)
    cell.add_view(overview(world, width=640, height=360))
    traj = json.loads(gzip.decompress((ROOT / "cells/tm_tray/plans/strict_cycle1.json.gz").read_bytes()))
    T_END = 10.0
    w = time.time()
    warm = cell.render("overview")                    # starts the cell, first frame (shaders)
    r["t_first_frame"] = round(time.time() - w, 1)
    r["first_frame"] = None if warm is None else {"shape": list(warm.shape), "max": int(warm.max())}
    a = cell.run(traj, t_end=T_END)
    frames, times = [], []

    def hook(t, *_):
        if t >= 0 and (not frames or t - frames[-1][0] >= 0.5 - 1e-9):
            s = time.time()
            img = cell.render("overview")
            times.append(time.time() - s)
            frames.append((t, None if img is None else img.copy()))
    b = cell.run(traj, t_end=T_END, on_sample=hook)
    try:
        r["usd_lag_m"] = round(cell.render_lag(), 5)          # at the end of the rendered run
    except Exception as e:  # noqa: BLE001
        r["usd_lag_m"] = repr(e)
    c = cell.run(traj, t_end=T_END)
    imgs = [f for _, f in frames if f is not None]
    r["frames"] = len(frames)
    r["none_frames"] = sum(f is None for _, f in frames)
    r["blank_frames"] = sum(int(f[::8, ::8].max()) < 8 for f in imgs)
    r["s_per_render"] = round(float(np.median(times)), 4) if times else None
    spread = float(np.abs(a.q - c.q).max())
    r["q_spread_without_render_rad"] = spread
    r["q_change_with_render_rad"] = float(np.abs(a.q - b.q).max())
    # PhysX is float32: an app update may round the state by an ULP (6e-8 rad at 0.5 rad,
    # Isaac Sim 5.0); more than 1e-6 rad (1 um at 1 m) would be rendering changing the run
    r["run_unchanged"] = r["q_change_with_render_rad"] <= max(1e-6, 2.0 * spread)
    if imgs:
        first, last = imgs[0], imgs[-1]
        r["mean_first"], r["mean_last"] = round(float(first.mean()), 1), round(float(last.mean()), 1)
        d = np.abs(last.astype(int) - first.astype(int)).max(axis=2)
        r["moved"] = round(float(d.mean()), 2)
        r["changed_pct"] = round(100.0 * float((d > 10).mean()), 2)
        r["unchanged"] = sum(np.array_equal(x, y) for x, y in zip(imgs, imgs[1:]))
        try:
            from PIL import Image
            for tag, img in (("first", first), ("last", last)):
                p = out_dir / f"live_probe_{mode}_{tag}.png"
                Image.fromarray(img).save(p)
                r[f"png_{tag}"] = str(p)
        except Exception as e:  # noqa: BLE001
            r["png_error"] = repr(e)
    lag = r.get("usd_lag_m")
    r["ok"] = bool(imgs and r["blank_frames"] == 0 and r.get("changed_pct", 0) > 1.0 and r["run_unchanged"]
                   and (not isinstance(lag, float) or lag < 0.005))
    r["t_total"] = round(time.time() - t0, 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="play,play-nosync,step")
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

"""Can Isaac Sim render here (RTX)? One RGB frame of a compiled stage (or a test scene)
from a camera looking at the robot, saved as PNG; JSON verdict with timings.

    OMNI_KIT_ACCEPT_EULA=YES python tools/isaac/render_probe.py --usd reports/isaac/strict.usda \
        --png reports/isaac/render.png --out reports/isaac/render_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--png", default="reports/isaac/render.png")
    ap.add_argument("--out", default="reports/isaac/render_probe.json")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 960))
    a = ap.parse_args()
    out, t0 = {"ok": False}, time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "width": a.size[0], "height": a.size[1]})
        out["t_app"] = round(time.time() - t0, 1)
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux
        ctx = omni.usd.get_context()
        ctx.open_stage(str(Path(a.usd).resolve()))
        app.update()
        st = ctx.get_stage()
        UsdLux.DomeLight.Define(st, "/World/probe_dome").CreateIntensityAttr(1000.0)
        UsdLux.DistantLight.Define(st, "/World/probe_sun").CreateIntensityAttr(2500.0)
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(st.GetPrimAtPath("/World/robot"))
        c = bb.ComputeAlignedRange().GetMidpoint()
        look = (float(c[0]), float(c[1]), float(c[2]))
        cam = rep.create.camera(position=(look[0] + 2.2, look[1] - 2.2, look[2] + 1.4), look_at=look)
        rp = rep.create.render_product(cam, tuple(a.size))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        t1 = time.time()
        for _ in range(3):
            rep.orchestrator.step()
        img = np.asarray(ann.get_data())
        out["t_first_frames"] = round(time.time() - t1, 1)
        t2 = time.time()
        for _ in range(10):
            rep.orchestrator.step()
        img = np.asarray(ann.get_data())
        out["s_per_frame"] = round((time.time() - t2) / 10, 3)
        out["shape"] = list(img.shape)
        out["mean"] = float(img[..., :3].mean()) if img.size else None
        out["ok"] = bool(img.size and img[..., :3].std() > 1.0)       # not blank
        if img.size:
            from PIL import Image
            Image.fromarray(img[..., :3].astype("uint8")).save(a.png)
        app.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        out["trace"] = traceback.format_exc()
    out["t_total"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("[render]", json.dumps({k: v for k, v in out.items() if k != "trace"}), flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

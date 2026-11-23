"""One RTX frame of a USD stage (or of a small demo scene) as PNG, with a JSON verdict.

    python $ISAAC_KIT/render.py --usd scene.usda [--look-at /World/robot] --png out/x.png --out out/x.json
    python $ISAAC_KIT/render.py --demo --png out/demo.png            # no stage needed

The camera looks at the bounding box of --look-at (default: the whole stage) from
--direction, at a distance that fits the box. A dome and a distant light are added (most
authored stages have none). The verdict is ok when the image is not blank.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaac_util import app, report  # noqa: E402


def demo_stage(st):
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")
    g = UsdGeom.Cube.Define(st, "/World/ground")
    g.CreateSizeAttr(1.0)                     # UsdGeom.Cube defaults to size 2
    g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    g.AddScaleOp().Set(Gf.Vec3f(2, 2, 0.1))
    for i, (name, geom, pos, rgb) in enumerate([("box", UsdGeom.Cube, (0, 0, 0.05), (0.8, 0.3, 0.2)),
                                                 ("ball", UsdGeom.Sphere, (0.4, 0.2, 0.15), (0.2, 0.5, 0.8))]):
        p = geom.Define(st, f"/World/{name}")
        p.GetPrim().GetAttribute("size" if name == "box" else "radius").Set(1.0)
        p.AddTranslateOp().Set(Gf.Vec3d(*pos))
        p.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.1) if name == "box" else Gf.Vec3f(0.15, 0.15, 0.15))
        m = UsdShade.Material.Define(st, f"/World/looks/{name}")
        sh = UsdShade.Shader.Define(st, f"/World/looks/{name}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4 + 0.3 * i)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--look-at", default="/", help="prim whose bounding box the camera frames")
    ap.add_argument("--direction", type=float, nargs=3, default=(1.0, -1.0, 0.6), help="camera direction from the target")
    ap.add_argument("--png", default="reports/isaac/render.png")
    ap.add_argument("--out", default="reports/isaac/render.json")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 960))
    ap.add_argument("--frames", type=int, default=3, help="frames before the saved one (RTX converges)")
    a = ap.parse_args()
    if not a.usd and not a.demo:
        ap.error("--usd or --demo")
    out, t0 = {"ok": False}, time.time()
    try:
        sim = app(width=a.size[0], height=a.size[1])
        out["t_app"] = round(time.time() - t0, 1)
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Usd, UsdGeom, UsdLux
        ctx = omni.usd.get_context()
        if a.usd:
            ctx.open_stage(str(Path(a.usd).resolve()))
        else:
            ctx.new_stage()
        sim.update()
        st = ctx.get_stage()
        if a.demo:
            demo_stage(st)
        target = st.GetPrimAtPath(a.look_at if a.look_at != "/" else "/")
        if not target:
            raise ValueError(f"no prim {a.look_at}")
        UsdLux.DomeLight.Define(st, "/render_dome").CreateIntensityAttr(1000.0)
        UsdLux.DistantLight.Define(st, "/render_sun").CreateIntensityAttr(2500.0)
        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(target).ComputeAlignedRange()
        c, size = rng.GetMidpoint(), rng.GetSize()
        r = max(0.5 * math.sqrt(sum(float(s) ** 2 for s in size)), 0.05)
        d = np.asarray(a.direction, float); d /= np.linalg.norm(d)
        dist = r / math.tan(math.radians(25)) * 1.1          # fits a sphere of radius r in a ~50 deg view
        look = tuple(float(v) for v in c)
        out["look_at"], out["radius"] = look, round(r, 3)
        cam = rep.create.camera(position=tuple(float(v) for v in (np.asarray(look) + dist * d)), look_at=look)
        rp = rep.create.render_product(cam, tuple(a.size))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        t1 = time.time()
        for _ in range(max(1, a.frames)):
            rep.orchestrator.step()
        img = np.asarray(ann.get_data())
        out["t_first_frames"] = round(time.time() - t1, 1)       # ~600 s on a T4 with a cold shader cache
        t2 = time.time()
        rep.orchestrator.step()
        img = np.asarray(ann.get_data())
        out["s_per_frame"] = round(time.time() - t2, 3)
        out["shape"] = list(img.shape)
        out["ok"] = bool(img.size and img[..., :3].std() > 1.0)
        if img.size:
            from PIL import Image
            Path(a.png).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img[..., :3].astype("uint8")).save(a.png)
            out["png"] = a.png
        report(out, a.out, "render")             # before close: closing the app ends the process
        sim.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        out["trace"] = traceback.format_exc()
        print(out["trace"], flush=True)
    report(out, a.out, "render")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

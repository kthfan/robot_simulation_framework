"""Does Isaac physics work here? Headless start, a box falling on a ground box (or a given
stage stepped for one second), and the physics tensor API. JSON verdict.

    python $ISAAC_KIT/probe.py [--usd scene.usda] [--out reports/isaac/probe.json]

With --usd it lists the articulations, their DOFs and links, and the rigid bodies.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaac_util import app, report, start_physics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd")
    ap.add_argument("--out", default="reports/isaac/probe.json")
    ap.add_argument("--dt", type=float, default=1e-3)
    a = ap.parse_args()
    out, t0 = {"ok": False}, time.time()
    try:
        sim = app()
        out["t_app"] = round(time.time() - t0, 1)
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics
        try:
            import isaacsim.core.version as v
            out["isaac_version"] = v.get_version()
        except Exception as e:  # noqa: BLE001
            out["isaac_version"] = repr(e)
        ctx = omni.usd.get_context()
        ctx.open_stage(str(Path(a.usd).resolve())) if a.usd else ctx.new_stage()
        sim.update()
        st = ctx.get_stage()
        if not a.usd:
            UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(st, 1.0)
            UsdPhysics.Scene.Define(st, "/physicsScene").CreateGravityMagnitudeAttr(9.81)
            g = UsdGeom.Cube.Define(st, "/ground")
            g.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, -0.05))   # translate, then scale:
            g.AddScaleOp().Set(Gf.Vec3f(2, 2, 0.1))                                        # top face at z = 0
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            b = UsdGeom.Cube.Define(st, "/box")
            b.CreateSizeAttr(0.1)
            b.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
            UsdPhysics.CollisionAPI.Apply(b.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(b.GetPrim())
            UsdPhysics.MassAPI.Apply(b.GetPrim()).CreateMassAttr(1.0)
        out["prims"] = len(list(st.Traverse()))
        physx, _, view = start_physics(ctx, a.dt)
        n = int(round(1.0 / a.dt))
        w = time.time()
        for k in range(n):
            physx.update_simulation(a.dt, k * a.dt)
        out["steps"], out["wall_s"] = n, round(time.time() - w, 2)
        if a.usd:
            arts = [p.GetPath().pathString for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            out["articulations"] = {}
            for path in arts:
                av = view.create_articulation_view(path)
                out["articulations"][path] = {"dof": int(av.max_dofs), "dof_names": list(av.shared_metatype.dof_names),
                                              "links": list(av.shared_metatype.link_names),
                                              "q": av.get_dof_positions().tolist()}
            out["rigid_bodies"] = sum(1 for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
            out["ok"] = True
        else:
            z = float(view.create_rigid_body_view("/box").get_transforms()[0, 2])
            out["box_z"] = round(z, 4)
            out["ok"] = abs(z - 0.05) < 0.005            # resting on the ground
        report(out, a.out, "probe")
        sim.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        out["trace"] = traceback.format_exc()
        print(out["trace"], flush=True)
    report(out, a.out, "probe")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

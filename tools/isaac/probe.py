"""Can Isaac Sim run here? Headless start, a box falling on the ground, and the
physics tensor API (the pieces ``rsf.sim.isaac_cell`` uses). Prints a JSON verdict.

    OMNI_KIT_ACCEPT_EULA=YES python tools/isaac/probe.py [--usd world.usda] [--out reports/isaac_probe.json]

With ``--usd`` it also opens a compiled world (``rsf.compile.usd``) and steps it.
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
    ap.add_argument("--usd")
    ap.add_argument("--out", default="reports/isaac_probe.json")
    a = ap.parse_args()
    out = {"ok": False, "stages": {}}
    t0 = time.time()

    def stage(name, **kw):
        out["stages"][name] = {"t": round(time.time() - t0, 1), **kw}
        print(f"[probe] {name} {kw}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))

    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        stage("app")
        import omni.usd
        import omni.physx
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
        try:
            import isaacsim.core.version as v
            out["isaac_version"] = v.get_version()
        except Exception as e:  # noqa: BLE001
            out["isaac_version"] = repr(e)
        ctx = omni.usd.get_context()
        if a.usd:
            ctx.open_stage(str(Path(a.usd).resolve()))
        else:
            ctx.new_stage()
        app.update()
        st = ctx.get_stage()
        if not a.usd:
            UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(st, 1.0)
            UsdPhysics.Scene.Define(st, "/physicsScene").CreateGravityMagnitudeAttr(9.81)
            g = UsdGeom.Cube.Define(st, "/ground")
            g.CreateSizeAttr(1.0)
            g.AddScaleOp().Set(Gf.Vec3f(2, 2, 0.1))
            g.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "pre").Set(Gf.Vec3d(0, 0, -0.05))
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            b = UsdGeom.Cube.Define(st, "/box")
            b.CreateSizeAttr(0.1)
            b.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
            UsdPhysics.CollisionAPI.Apply(b.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(b.GetPrim())
            UsdPhysics.MassAPI.Apply(b.GetPrim()).CreateMassAttr(1.0)
        stage("stage", prims=len(list(st.Traverse())))
        phys = omni.physx.get_physx_interface()
        phys.start_simulation()
        dt, n = 1e-3, 1000
        w = time.time()
        for k in range(n):
            phys.update_simulation(dt, k * dt)
        stage("step", steps=n, wall_s=round(time.time() - w, 2))
        import omni.physics.tensors as T
        view = T.create_simulation_view("numpy")
        info = {}
        if a.usd:
            arts = [p.GetPath().pathString for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            info["articulations"] = arts
            if arts:
                av = view.create_articulation_view(arts[0])
                info["dof"] = int(av.max_dofs)
                info["dof_names"] = list(av.shared_metatype.dof_names)
                info["links"] = list(av.shared_metatype.link_names)
                info["q"] = av.get_dof_positions().tolist()
        else:
            rv = view.create_rigid_body_view("/box")
            info["box_z"] = float(rv.get_transforms()[0, 2])
        stage("tensors", **info)
        out["ok"] = True
        stage("done")
        app.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        out["trace"] = traceback.format_exc()
        stage("error")
        print(out["trace"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

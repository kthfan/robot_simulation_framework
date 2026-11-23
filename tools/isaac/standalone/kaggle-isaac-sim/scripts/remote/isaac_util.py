"""Small helpers for Isaac Sim scripts run by isaac_kaggle.sh (on Kaggle: ``import isaac_util``;
``$ISAAC_KIT`` is on ``PYTHONPATH``).

    from isaac_util import app, start_physics, pose7_to_T, report
    sim = app()                       # SimulationApp first, before any omni import
    import omni.usd
    ctx = omni.usd.get_context(); ctx.open_stage("scene.usda"); sim.update()
    physx, sim_if, view = start_physics(ctx, 1e-3)
    for k in range(n): physx.update_simulation(1e-3, k * 1e-3)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
_APP = None
_T0 = time.time()


def app(headless: bool = True, **config):
    """The one ``SimulationApp`` of this process. Create it before importing ``omni.*`` or
    Isaac's ``pxr``. ``app().close()`` ends the Python process: write results before it."""
    global _APP
    if _APP is None:
        from isaacsim import SimulationApp
        _APP = SimulationApp({"headless": headless, **config})
    return _APP


def start_physics(ctx, dt: float):
    """Start PhysX on the context's stage and a numpy tensor view of it, the way Isaac's own
    ``SimulationContext.initialize_physics`` does. Returns ``(physx, simulation_interface, view)``.

    The tensor view needs the stage id; without it, ``create_simulation_view`` fails with
    "Failed to get a valid attached USD stage id"."""
    import omni.physics.tensors as tensors
    import omni.physx
    physx = omni.physx.get_physx_interface()
    sim_if = omni.physx.get_physx_simulation_interface()
    physx.force_load_physics_from_usd()
    physx.start_simulation()
    physx.update_simulation(dt, 0.0)
    view = tensors.create_simulation_view("numpy", stage_id=ctx.get_stage_id())
    view.set_subspace_roots("/")
    physx.update_simulation(dt, 0.0)
    return physx, sim_if, view


def pose7_to_T(x):
    """Tensor API pose ``[px, py, pz, qx, qy, qz, qw]`` (quaternion xyzw) -> 4x4 numpy."""
    import numpy as np
    x = np.asarray(x, float)
    qx, qy, qz, qw = x[3:7]
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, x[:3]
    return T


def report(out: dict, path: str | os.PathLike, tag: str = "isaac") -> None:
    """Write ``out`` as JSON (with total time) and print it: call it before ``app().close()``."""
    out["t_total"] = round(time.time() - _T0, 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1, default=str))
    print(f"[{tag}]", json.dumps({k: v for k, v in out.items() if k != "trace"}, default=str), flush=True)

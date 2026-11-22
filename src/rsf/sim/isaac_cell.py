"""Isaac Sim replay of a planned run (P5): the same cell and the same report as
:class:`rsf.sim.mujoco_cell.MujocoCell`, with NVIDIA PhysX.

The stage is :func:`rsf.compile.usd.world_stage` of the world (objects only where
the run starts them), so the planner, MuJoCo and Isaac see one cell. The servo is
the MuJoCo cell's: the arm gravity compensated (its links ignore gravity), PD in
the joint drives (position and velocity targets from :class:`~rsf.sim.common.Reference`),
the integral term as an applied joint force; jaws are force-limited position
drives. :meth:`IsaacCell.run` returns an :class:`~rsf.sim.common.ExecReport`, judged
by :mod:`rsf.sim.judge` as the MuJoCo run is.

Needs Isaac Sim (``pip install isaacsim[all,extscache]``, Python 3.12 for 6.x, an
RTX-class GPU and driver; ``OMNI_KIT_ACCEPT_EULA=YES``). Only the low-level,
long-stable interfaces are used: ``omni.physx`` to step physics (no rendering)
and read contact reports, ``omni.physics.tensors`` (numpy) for the articulation
and the objects. Nothing is imported until a cell is made, so this module
imports anywhere.

PhysX and MuJoCo differ in contact (rigid, with a contact offset, vs soft with
``solref``/``solimp``) and friction (static and dynamic, patch friction, vs one
coefficient, elliptic cones); :class:`IsaacParams` keeps the shared physics of
:class:`~rsf.sim.common.SimParams` and adds PhysX's own solver settings.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from rsf.core import se3
from rsf.sim.common import ExecReport, Reference, SimParams


@dataclass
class IsaacParams(SimParams):
    solver: str = "TGS"
    position_iterations: int = 32
    velocity_iterations: int = 4
    gpu_dynamics: bool = False         #: CPU PhysX: one cell, deterministic, no GPU pipeline needed
    contact_offset: float = 0.002      #: m: contacts generated within this distance
    min_link_mass: float = 0.01        #: kg for massless links (PhysX needs mass on every body)
    arm_max_torque: float = 1.0e4      #: N m: drive limit (MuJoCo's servo is unlimited)
    max_joint_velocity: float = 1.0e6  #: deg/s or m/s: PhysX joint speed limit (off)
    headless: bool = True


_APP = None


def app(headless: bool = True):
    """The (one) Isaac Sim application, started on first use."""
    global _APP
    if _APP is None:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaacsim import SimulationApp
        _APP = SimulationApp({"headless": headless})
    return _APP


def start_physics(ctx, dt: float):
    """Start PhysX on the context's stage and a numpy tensor view of it, as Isaac's own
    ``SimulationContext.initialize_physics`` does: ``(physx, simulation interface, view)``."""
    import omni.physics.tensors as tensors
    import omni.physx
    physx = omni.physx.get_physx_interface()
    sim_if = omni.physx.get_physx_simulation_interface()
    stage_id = ctx.get_stage_id()
    physx.force_load_physics_from_usd()
    physx.start_simulation()
    physx.update_simulation(dt, 0.0)
    view = tensors.create_simulation_view("numpy", stage_id=stage_id)
    view.set_subspace_roots("/")
    physx.update_simulation(dt, 0.0)
    return physx, sim_if, view


def _T7(x) -> np.ndarray:
    """``[px, py, pz, qx, qy, qz, qw]`` -> 4x4."""
    x = np.asarray(x, float)
    return se3.make_T(se3.quat_to_matrix((x[6], x[3], x[4], x[5])), x[:3])


class IsaacCell:
    def __init__(self, world, occupancy: Optional[dict] = None, params: Optional[IsaacParams] = None,
                 flange: str = "flange", usd_path=None):
        import importlib.util
        if importlib.util.find_spec("isaacsim") is not None:
            app(self.p_headless(params))                   # Isaac's own pxr loads with the app
        from rsf.compile import usd as U
        from rsf.model.urdf import UrdfModel
        self.world = world
        self.p = params or IsaacParams()
        objs = world.objects if occupancy is None else tuple(o for o in world.objects if o.at in occupancy)
        self.object_names = [o.name for o in objs]
        self.flange = flange
        self.usd_path = Path(usd_path) if usd_path else Path(tempfile.mkdtemp(prefix="rsf_isaac_")) / "cell.usda"
        self.usd_path.parent.mkdir(parents=True, exist_ok=True)
        stage = U.world_stage(replace(world, objects=objs), self.usd_path, params=self.p)
        self.links = U.link_map(stage)                     # link -> (body, T_body_link)
        self.urdf = world.robot.urdf if isinstance(world.robot.urdf, UrdfModel) else UrdfModel.from_file(world.robot.urdf)
        self.base = np.asarray(world.robot.base, float)
        self._names = {}                                    # prim path -> (kind, name, body)
        for prim in stage.Traverse():
            k = prim.GetCustomDataByKey("rsf:kind")
            if k:
                self._names[str(prim.GetPath())] = (k, prim.GetCustomDataByKey("rsf:name"),
                                                    prim.GetCustomDataByKey("rsf:object"))
        self.object_paths = {o.name: f"{U.ROOT}/objects/{U.prim_name(o.name)}" for o in objs}
        self.robot_path = f"{U.ROOT}/robot"
        self._prim_name = U.prim_name
        self._started = False

    @staticmethod
    def p_headless(params) -> bool:
        return True if params is None else params.headless

    # ── Isaac side ─────────────────────────────────────────────────────────
    def _start(self):
        app(self.p.headless)
        import omni.usd
        ctx = omni.usd.get_context()
        ctx.open_stage(str(self.usd_path))
        _APP.update()
        self.physx, self.sim_if, self.view = start_physics(ctx, self.p.timestep)
        self.art = self.view.create_articulation_view(self.robot_path)
        meta = self.art.shared_metatype
        self.dof_names = list(meta.dof_names)
        self.link_names = list(meta.link_names)
        self.objs = {n: self.view.create_rigid_body_view(p) for n, p in self.object_paths.items()}
        self._idx = np.arange(1, dtype=np.int32)
        self._started = True

    def _path_name(self, i: int):
        from pxr import PhysicsSchemaTools
        return str(PhysicsSchemaTools.intToSdfPath(i))

    def _who(self, path: str):
        """``(is_robot, name)`` for a collider or body path."""
        while path and path not in self._names:
            path = path.rsplit("/", 1)[0]
        kind, name, obj = self._names.get(path, ("", path, None))
        if kind.startswith("robot"):
            return True, name
        if kind == "object_solid":
            return False, obj
        return False, name

    def _contacts(self, t, out):
        headers, data = self.sim_if.get_contact_report()
        for h in headers:
            c0 = self._path_name(getattr(h, "collider0", h.actor0))
            c1 = self._path_name(getattr(h, "collider1", h.actor1))
            r0, n0 = self._who(c0)
            r1, n1 = self._who(c1)
            if r0 == r1:
                continue
            sep = min((data[h.contact_data_offset + k].separation for k in range(h.num_contact_data)), default=0.0)
            if sep >= -self.p.contact_tol:
                continue
            link, other = (n0, n1) if r0 else (n1, n0)
            out.append({"t": round(t, 4), "link": link, "other": other, "depth": float(-sep)})

    def _flange_actual(self, links_T):
        body, T_bl = self.links[self.flange]
        return _T7(links_T[self.link_names.index(self._prim_name(body))]) @ T_bl

    # ── the run ────────────────────────────────────────────────────────────
    def run(self, traj: dict, t_end: Optional[float] = None, progress=None, on_sample=None) -> ExecReport:
        """Replay a ``rsf.trajectory/1`` dict; *on_sample(t, cell)* at every sample."""
        if not self._started:
            self._start()
        p = self.p
        ref = Reference(traj)
        names, T, Q = ref.names, ref.T, ref.Q
        col = np.array([names.index(n) for n in self.dof_names])     # trajectory column of each DOF
        arm = np.array([self.urdf.joints[n].type != "prismatic" for n in self.dof_names])
        kp = np.zeros(len(self.dof_names))
        kp[arm] = np.array(p.arm_kp[:int(arm.sum())]) * p.arm_ki_per_s
        art, idx = self.art, self._idx
        art.set_dof_positions(Q[0][col][None], idx)
        art.set_dof_velocities(np.zeros((1, len(col))), idx)
        art.set_dof_position_targets(Q[0][col][None], idx)
        t_stop = (T[-1] if t_end is None else min(t_end, T[-1])) + p.settle
        n_steps = int(math.ceil(t_stop / p.timestep))
        every_s = max(1, int(round(p.sample_dt / p.timestep)))
        every_c = max(1, int(round(p.contact_dt / p.timestep)))
        ts, fl, fp, qs, contacts = [], [], [], [], []
        objs = {n: [] for n in self.object_names}
        integ = np.zeros(len(col))
        tool = self.flange
        wall = time.time()
        for k in range(n_steps + 1):
            t = k * p.timestep - p.settle
            q_ref, qd_ref = ref(t)
            qr, qdr = q_ref[col], qd_ref[col]
            q = art.get_dof_positions()[0]
            e = np.where(arm, qr - q, 0.0)
            integ += e * p.timestep
            art.set_dof_position_targets(qr[None], idx)
            art.set_dof_velocity_targets(np.where(arm, qdr, 0.0)[None], idx)
            art.set_dof_actuation_forces((kp * integ)[None], idx)
            if k % every_c == 0 and t >= 0:
                self._contacts(t, contacts)
            if k % every_s == 0 and t >= 0:
                ts.append(t)
                fl.append(self._flange_actual(art.get_link_transforms()[0]))
                for n, v in self.objs.items():
                    objs[n].append(_T7(v.get_transforms()[0]))
                fp.append(self.base @ self.urdf.link_pose(tool, dict(zip(names, q_ref))))
                qa = np.array(q_ref, float)
                qa[col] = q
                qs.append(qa)
                if on_sample is not None:
                    on_sample(t, self)
                if progress and len(ts) % 500 == 0:
                    progress(t, time.time() - wall)
            self.physx.update_simulation(p.timestep, k * p.timestep)
        return ExecReport(np.array(ts), np.array(fl), np.array(fp), np.array(qs),
                          {n: np.array(v) for n, v in objs.items()}, contacts, list(traj.get("segments", [])),
                          {"backend": "isaac", "params": dict(vars(p)), "usd": str(self.usd_path),
                           "wall_s": round(time.time() - wall, 1), "steps": n_steps})

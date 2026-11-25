"""MuJoCo replay of a planned run (P5): the planned cell, with physics for the objects.

The model is :func:`rsf.compile.mjcf.world_spec` of the world, with objects only
where the run starts them. The arm is gravity compensated and follows the
trajectory with a position + velocity servo per joint (the real controller
tracks a joint stream; a PID, so a carried load leaves no steady error); the jaws are position servos with a force limit.
Objects are free bodies with friction and mass; nothing holds them but contact.

Robot-robot contact is off (self-collision is the planner's job, P2-P4); robot
vs everything else and object vs everything else is on. MuJoCo combines the
friction of a pair by the max of the two geoms, so with the robot's geoms at
``friction["robot"]`` and the objects' below it, the plate-tray pair gets
exactly the robot value (D30: plate 0.5 static / 0.4 dynamic, combine max;
MuJoCo has one sliding coefficient, the dynamic value is used).

:meth:`MujocoCell.run` returns an execution report (backend independent, read
by :mod:`rsf.sim.judge`): samples of the flange (actual and planned), every
object's pose, and the robot's contacts with penetration depth.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Optional

import numpy as np

from rsf.compile.mjcf import ROBOT_PREFIX, world_spec
from rsf.sim.common import ARM_KP, ARM_KV, ExecReport, Reference, SimParams, T_of  # noqa: F401

_T = T_of

#: Colours of rendered views (the compiled model's own are for collision debugging).
COLORS = {"robot": (0.62, 0.64, 0.68, 1), "gripper": (0.25, 0.27, 0.30, 1), "tray": (0.93, 0.55, 0.15, 1),
          "jig": (0.30, 0.50, 0.80, 1), "static": (0.82, 0.82, 0.80, 1), "floor": (0.55, 0.56, 0.58, 1)}


def style(m, objects) -> None:
    """Colour the geoms of a cell model by role and light it for pictures (visual only)."""
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        n = m.geom(g).name
        if b.startswith(ROBOT_PREFIX):
            c = "gripper" if any(k in b for k in ("gripper", "flange")) else "robot"
        elif b in objects:
            c = "tray"
        elif n.startswith("jig/"):
            c = "jig"
        elif n == "floor":
            c = "floor"
        else:
            c = "static"
        m.geom_rgba[g] = COLORS[c]
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.55


class MujocoCell:
    def __init__(self, world, occupancy: Optional[dict] = None, params: Optional[SimParams] = None,
                 flange: str = "flange"):
        import mujoco
        self.mj = mujoco
        self.world = world
        self.p = params or SimParams()
        objs = world.objects if occupancy is None else tuple(o for o in world.objects if o.at in occupancy)
        self.object_names = [o.name for o in objs]
        spec = world_spec(replace(world, objects=objs), timestep=self.p.timestep, actuators=False)
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = self.p.impratio
        spec.option.noslip_iterations = self.p.noslip_iterations
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        fr = self.p.friction
        for b in spec.bodies:
            robot = b.name.startswith(ROBOT_PREFIX)
            obj = b.name in self.object_names
            if robot:
                b.gravcomp = 1.0
            for g in b.geoms:
                if robot:
                    g.contype, g.conaffinity, f = 2, 1, fr["robot"]
                else:
                    g.contype, g.conaffinity, f = 1, 3, fr["object" if obj else "static"]
                g.friction = [f, 0.005, 0.0001]
                g.solref = list(self.p.solref)
                g.solimp = list(self.p.solimp) + [0.5, 2.0]
        self.arm, self.jaws = [], []
        for j in spec.joints:
            if not j.name.startswith(ROBOT_PREFIX) or j.type not in (mujoco.mjtJoint.mjJNT_HINGE,
                                                                    mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            j.armature = self.p.armature if j.type == mujoco.mjtJoint.mjJNT_HINGE else self.p.jaw_armature
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                k = len(self.arm)
                a = spec.add_actuator(name=f"{j.name}/pos", target=j.name)
                a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                a.set_to_position(kp=self.p.arm_kp[k])
                v = spec.add_actuator(name=f"{j.name}/vel", target=j.name)
                v.trntype = mujoco.mjtTrn.mjTRN_JOINT
                v.set_to_velocity(kv=self.p.arm_kv[k])
                self.arm.append(j.name)
            else:
                a = spec.add_actuator(name=f"{j.name}/pos", target=j.name)
                a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                a.set_to_position(kp=self.p.jaw_kp, kv=2.0 * math.sqrt(self.p.jaw_kp * self.p.jaw_armature))
                a.forcelimited = True
                a.forcerange = [-self.p.jaw_force, self.p.jaw_force]
                self.jaws.append(j.name)
        self.spec = spec
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.d_plan = mujoco.MjData(self.m)
        m = self.m
        self.flange_id = m.body(ROBOT_PREFIX + flange).id
        self.robot_body = np.array([m.body(i).name.startswith(ROBOT_PREFIX) for i in range(m.nbody)])
        self.obj_id = {n: m.body(n).id for n in self.object_names}

    # ── views: pictures of the run (visual only; the physics is not touched) ──
    def add_view(self, view) -> None:
        """A camera :meth:`render` can draw (:class:`rsf.sim.common.View`)."""
        mj, m = self.mj, self.m
        if not hasattr(self, "_views"):
            self._views, self._renderers = {}, {}
            self.d_view = mj.MjData(m)
            style(m, set(self.object_names))
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(view.width))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(view.height))
        self._views[view.name] = view
        self._renderers.pop(view.name, None)

    def render(self, name: str) -> np.ndarray:
        """The view *name* of the current state, ``(h, w, 3)`` uint8 RGB."""
        mj, m = self.mj, self.m
        v = self._views[name]
        r = self._renderers.get(name)
        if r is None:
            r = self._renderers[name] = mj.Renderer(m, int(v.height), int(v.width))
        dv = self.d_view                          # fresh kinematics of the state (d.xpos lags a step)
        dv.qpos[:] = self.d.qpos
        mj.mj_kinematics(m, dv)
        mj.mj_camlight(m, dv)
        eye, target = np.asarray(v.eye, float), np.asarray(v.target, float)
        f = target - eye
        dist = float(np.linalg.norm(f))
        f /= dist
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance = dist
        cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
        cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
        m.vis.global_.fovy = float(v.fov_deg)
        r.update_scene(dv, cam)
        return r.render()

    def move_view(self, name: str, eye, target, fov_deg=None) -> None:
        """Move the view *name*'s camera (its size stays)."""
        from dataclasses import replace as _r
        v = self._views[name]
        self._views[name] = _r(v, eye=tuple(map(float, eye)), target=tuple(map(float, target)),
                               fov_deg=float(v.fov_deg if fov_deg is None else fov_deg))

    def close_views(self) -> None:
        for r in getattr(self, "_renderers", {}).values():
            r.close()
        self._renderers = {}

    # ── helpers ────────────────────────────────────────────────────────────
    def _qadr(self, names):
        return np.array([self.m.joint(ROBOT_PREFIX + n).qposadr[0] for n in names])

    def _set_robot(self, d, adr, q):
        d.qpos[adr] = q

    def _contacts(self, t, out):
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -self.p.contact_tol:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = self.robot_body[b1], self.robot_body[b2]
            if r1 == r2:
                continue
            gr, go = (c.geom1, c.geom2) if r1 else (c.geom2, c.geom1)
            out.append({"t": round(t, 4), "link": m.body(m.geom_bodyid[gr]).name[len(ROBOT_PREFIX):],
                        "other": m.body(m.geom_bodyid[go]).name if m.geom_bodyid[go] else m.geom(go).name,
                        "depth": float(-c.dist)})

    # ── the run ────────────────────────────────────────────────────────────
    def run(self, traj: dict, t_end: Optional[float] = None, progress=None, on_sample=None) -> ExecReport:
        """Replay a ``rsf.trajectory/1`` dict (``t``, ``q``, ``qd``, ``joint_names``).

        *on_sample(t, model, data)* is called at every sample (e.g. to render)."""
        mj, m, d, p = self.mj, self.m, self.d, self.p
        ref = Reference(traj)
        names, T, Q = ref.names, ref.T, ref.Q
        adr = self._qadr(names)
        act_pos = np.array([m.actuator(f"{ROBOT_PREFIX}{n}/pos").id for n in names])
        arm_idx = [i for i, n in enumerate(names) if ROBOT_PREFIX + n in self.arm]
        act_vel = np.array([m.actuator(f"{ROBOT_PREFIX}{names[i]}/vel").id for i in arm_idx])
        mj.mj_resetData(m, d)
        d.qpos[adr] = Q[0]
        mj.mj_forward(m, d)
        t_stop = (T[-1] if t_end is None else min(t_end, T[-1])) + p.settle
        n_steps = int(math.ceil(t_stop / p.timestep))
        every_s = max(1, int(round(p.sample_dt / p.timestep)))
        every_c = max(1, int(round(p.contact_dt / p.timestep)))
        ts, fl, fp, qs, contacts = [], [], [], [], []
        objs = {n: [] for n in self.object_names}
        arm_adr = adr[arm_idx]
        arm_dof = np.array([m.joint(ROBOT_PREFIX + names[i]).dofadr[0] for i in arm_idx])
        ki = np.array(p.arm_kp[:len(arm_idx)]) * p.arm_ki_per_s
        integ = np.zeros(len(arm_idx))
        wall = time.time()
        for k in range(n_steps + 1):
            t = k * p.timestep - p.settle             # trajectory time (negative while settling)
            q_ref, qd_ref = ref(t)
            d.ctrl[act_pos] = q_ref
            d.ctrl[act_vel] = qd_ref[arm_idx]
            e = q_ref[arm_idx] - d.qpos[arm_adr]
            integ += e * p.timestep
            d.qfrc_applied[arm_dof] = ki * integ
            if k % every_c == 0 and t >= 0:
                self._contacts(t, contacts)
            if k % every_s == 0 and t >= 0:
                ts.append(t)
                # after mj_step, d.xpos belongs to the state before the step: fresh kinematics
                dp = self.d_plan
                dp.qpos[:] = d.qpos
                mj.mj_kinematics(m, dp)
                fl.append(_T(dp.xpos[self.flange_id], dp.xmat[self.flange_id]))
                for n, i in self.obj_id.items():
                    objs[n].append(_T(dp.xpos[i], dp.xmat[i]))
                dp.qpos[adr] = q_ref
                mj.mj_kinematics(m, dp)
                fp.append(_T(dp.xpos[self.flange_id], dp.xmat[self.flange_id]))
                qs.append(d.qpos[adr].copy())
                if on_sample is not None:
                    on_sample(t, m, d)
                if progress and len(ts) % 500 == 0:
                    progress(t, time.time() - wall)
            mj.mj_step(m, d)
        return ExecReport(np.array(ts), np.array(fl), np.array(fp), np.array(qs),
                          {n: np.array(v) for n, v in objs.items()}, contacts, list(traj.get("segments", [])),
                          {"backend": "mujoco", "params": {k: v for k, v in vars(p).items()},
                           "wall_s": round(time.time() - wall, 1), "steps": n_steps})

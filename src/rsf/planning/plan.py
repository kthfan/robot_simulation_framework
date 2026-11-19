"""P4: a workflow run as one timed joint trajectory.

The run is cut into segments:

* ``line`` / ``path``: the constrained arrivals of the teach sequences (inside
  shelves, the tilt). Re-walked here with the IK on a trapezoid timed to the
  segment's Cartesian caps; the same rules as the teach engine apply.
* ``free``: every other move (between sequences, between runs inside one, home).
  Planned by cuRobo on a GPU (``tools/kaggle/curobo_free_plan.py``) from a
  *job* this module writes, or by a CPU fallback; then retimed to the joint caps
  and, when an object is carried, to its Cartesian caps.
* ``hold``: the arm still while the jaws move (or a dwell); events are recorded.

``assemble`` checks every sample of every segment on the CPU with the exact
meshes and the segment's rules before it returns a trajectory: the GPU plan
is a proposal, the CPU check is the gate.

Process parameters, ``process.motion``::

    motion:
      dt: 0.01                       # output sample period (s)
      caps:                          # Cartesian caps of the tool (and a carried object)
        in_jig: {v: 0.05, a: 0.25, w: 0.25, alpha: 1.0}
        carry:  {v: 0.25, a: 1.5,  w: 0.5,  alpha: 3.0, a_down: 4.0}
        empty:  {v: 0.25, a: 1.5,  w: 0.5,  alpha: 3.0}
      joint: {v: 1.0, a: 2.0}        # free moves, rad/s, rad/s^2
      jaw: {speed: 0.05, settle: 0.5}
      home: [0, 0, 0, 0, 0, 0]
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from rsf.collision import ContactPolicy, ContactRule
from rsf.core import se3
from rsf.planning import timing as TM
from rsf.planning.cartesian import walk


# ─────────────────────────────────────────────────────────────────────────────
# Segments
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(eq=False)
class Segment:
    name: str
    kind: str                         #: line | path | free | hold
    joints: dict                      #: other joints (jaws) during the segment (start values for hold)
    scene: object = None              #: rsf.planning.teach.Scene for the collision check
    policy: Optional[ContactPolicy] = None
    poses: tuple = ()                 #: tool poses (line/path)
    q_start: Optional[np.ndarray] = None
    q_goal: Optional[np.ndarray] = None
    caps: str = "empty"               #: key into process.motion.caps
    carried: Optional[tuple] = None   #: (name, link, solids, T_link_object) when something is held
    duration: float = 0.0             #: hold
    joints_end: Optional[dict] = None #: hold: jaw targets
    event: str = ""
    # filled in later
    t: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None    #: (n, dof) arm joints
    J: Optional[np.ndarray] = None    #: (n, m) other joints


def _motion(process: dict) -> dict:
    m = dict(process.get("motion") or {})
    m.setdefault("dt", 0.01)
    m.setdefault("joint", {"v": 1.0, "a": 2.0})
    m.setdefault("jaw", {"speed": 0.05, "settle": 0.5})
    m.setdefault("caps", {})
    return m


def free_scene(world, occupancy: dict, exclude=()) -> tuple:
    """Static solids plus the object template at every occupied feature."""
    out = list(world.solids(objects=False))
    for key in occupancy:
        if key in exclude:
            continue
        obj = world.object_at(key)
        if obj is not None:
            out += obj.world_solids()
    return tuple(out)


def free_policy(margin: float) -> ContactPolicy:
    return ContactPolicy(margin, [ContactRule(allow=True, links=("link_0",), tags=("robot_seat",),
                                              label="base on its mount")])


def segments_for_workflow(world, workflow, teach: dict, *, margin: float = 0.005) -> list:
    """The run's segments, from the workflow and its teach table (``teach_workflow`` JSON).

    *teach* maps sequence name -> point list. Free segments get their start
    and goal joints, the scene of the cell at that moment of the run, and the
    carried object.
    """
    from rsf.model import registry
    from rsf.model.workflow import expand
    from rsf.planning.teach import Scene

    mo = _motion(world.process)
    ex = expand(workflow, world)
    occ = dict(workflow.initial_occupancy(world, workflow.resolve_sets(world)))
    home = np.array(mo.get("home", [0.0] * 6), dtype=float)
    segs, q, held, jaws = [], home, None, None
    n_free = 0

    def free_to(q_goal, joints, label):
        nonlocal n_free, q
        n_free += 1
        exclude = set()
        if held is not None:
            exclude = {held[4], held[5]}           # where it came from, where it goes
        sc = Scene(free_scene(world, occ, exclude), () if held is None else (held[:4],))
        segs.append(Segment(f"free{n_free:03d}:{label}", "free", dict(joints), sc, free_policy(margin),
                            q_start=np.array(q, float), q_goal=np.array(q_goal, float),
                            caps="carry" if held is not None else "empty",
                            carried=None if held is None else held[:4]))
        q = np.array(q_goal, float)

    for task in ex.tasks:
        if task.skill not in registry.names("skill"):
            segs.append(Segment(f"hold:{task.skill}", "hold", dict(jaws or {}), duration=0.0,
                                event=f"{task.skill} {task.args}"))
            continue
        for seq in registry.get("skill", task.skill)(world, task.args):
            pts = {p["name"]: p for p in teach[seq.name]}
            wps = seq.waypoints
            for i, wp in enumerate(wps):
                qw = np.array(pts[wp.name]["q"], float)
                if wp.arrive == "free":
                    free_to(qw, wp.joints, f"{seq.name}/{wp.name}")
                else:
                    segs.append(Segment(f"{seq.name}/{wp.name}", wp.arrive, dict(wp.joints),
                                        wp.along_scene or wp.scene, wp.along_policy or wp.policy,
                                        poses=tuple(wp.path) if wp.arrive == "path" else (wps[i - 1].T, wp.T),
                                        q_start=np.array(q, float), q_goal=qw,
                                        caps=wp.meta.get("caps", "empty"),
                                        carried=wp.scene.attached[0] if wp.scene.attached else None))
                    q = qw
                jaws = dict(wp.joints)
                if wp.action:
                    nxt = wps[i + 1].joints if i + 1 < len(wps) else wp.joints
                    segs.append(Segment(f"{seq.name}/{wp.name}:act", "hold", dict(wp.joints),
                                        joints_end=dict(nxt), event=wp.action))
                    jaws = dict(nxt)
            # what the sequence leaves in the hand
            last = wps[-1]
            if last.scene.attached:
                name, link, solids, T = last.scene.attached[0]
                key = seq.meta.get("slot")
                held = (name, link, solids, T, key, task.args.get("to"))
                occ.pop(key, None)
            else:
                if held is not None:
                    occ[seq.meta.get("slot")] = held[0]
                held = None
    free_to(home, jaws or {}, "home")
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Jobs for the GPU (cuRobo) and their results
# ─────────────────────────────────────────────────────────────────────────────
def write_job(world, segs: list, path, *, margin: float = 0.005, meta: Optional[dict] = None,
              root=None, arm=None) -> dict:
    """The free segments as a self-contained cuRobo planning job (JSON).

    Scenes (cuRobo dicts in the robot base frame) are stored once in
    ``scenes`` and referenced by id from each segment.

    *root*: the repository root; the robot URDF is written relative to it, so the
    job runs on another machine's checkout (Kaggle)."""
    from rsf.compile.curobo import scene_dict
    import math as _m
    import hashlib
    base = world.robot.base
    items, scenes = [], {}
    for s in segs:
        if s.kind != "free":
            continue
        sd = scene_dict(s.scene.solids, base, exclude=lambda x: "robot_seat" in x.tags)
        scene = hashlib.sha1(json.dumps(sd, sort_keys=True).encode()).hexdigest()[:12]
        scenes[scene] = sd                                   # segments share a few scenes
        carried = None
        if s.carried is not None:
            name, link, solids, T = s.carried
            carried = {"link": link, "boxes": [{"name": b.name, "dims": [float(v) for v in b.shape.size],
                                                "pose_in_link": (T @ b.T).tolist()} for b in solids]}
        bound = None
        if carried is not None and arm is not None:
            # the carry cone (carry_frame) as a cuRobo ToolPoseBound: the tool-frame axis
            # opposite gravity at the pick tilt stays within the cone of the base's vertical
            frame = carry_frame(arm, s, float(_motion(world.process).get("carry_max_tilt_deg", 15.0)))
            bound = {"tool_axis": [float(v) for v in -frame[0]], "reference_axis": [0.0, 0.0, 1.0],
                     "max_angle": math.acos(max(-1.0, min(1.0, frame[2])))}
        items.append({"name": s.name, "q_start": [float(v) for v in s.q_start], "q_goal": [float(v) for v in s.q_goal],
                      "lock_joints": {k: float(v) for k, v in s.joints.items()}, "scene": scene, "carried": carried,
                      "tool_bound": bound})
    import os
    urdf = Path(world.robot.urdf).resolve()
    rel = os.path.relpath(urdf, Path(root).resolve()) if root else str(urdf)
    job = {"schema": "rsf.curobo_job/1", "robot_urdf": rel,
           "base": base.tolist(), "tool_frame": world.robot.tool_frame,
           "limit_margin_deg": float(world.robot.meta.get("limit_margin_deg", 0.0)),
           "mesh_collision_margin": margin, "scenes": scenes, "segments": items, "meta": meta or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(job))
    return job


def attach_results(segs: list, results: dict) -> list:
    """Free segments' joint paths from a GPU result file (``{name: {success, q, dt}}``). Returns failures."""
    bad = []
    for s in segs:
        if s.kind != "free":
            continue
        r = results.get(s.name)
        if r is None or not r.get("success"):
            bad.append(s.name)
            continue
        Q = np.array(r["q"], float)
        Q[0], Q[-1] = s.q_start, s.q_goal
        s.Q = Q
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# Timing and the CPU gate
# ─────────────────────────────────────────────────────────────────────────────
def _carried_point(arm, q, carried):
    if carried is None:
        return None
    name, link, solids, T = carried
    c = np.mean([(T @ s.T)[:3, 3] for s in solids], axis=0)
    return arm.fk(q) @ np.append(c, 1.0)


def time_segment(arm, s: Segment, motion: dict) -> None:
    dt = float(motion["dt"])
    caps = TM.caps_from(motion["caps"][s.caps]) if s.caps in motion["caps"] else None
    if s.kind == "hold":
        j0, j1 = s.joints, s.joints_end or s.joints
        moves = [abs(j1[k] - j0.get(k, j1[k])) for k in j1] or [0.0]
        T = max(moves) / float(motion["jaw"]["speed"]) + (float(motion["jaw"]["settle"]) if max(moves) > 0 else 0.0)
        T = max(T, s.duration)
        n = max(1, math.ceil(T / dt)) if T > 0 else 1
        s.t = np.linspace(0.0, T, n + 1) if T > 0 else np.array([0.0])
        frac = np.clip(s.t / max(T - float(motion["jaw"]["settle"]), 1e-9), 0, 1) if T > 0 else np.ones(1)
        s.J = np.array([[j0.get(k, j1[k]) + (j1[k] - j0.get(k, j1[k])) * f for k in sorted(j1)] for f in frac]) if j1 else None
        return
    if s.kind in ("line", "path"):
        P = list(s.poses)
        if s.kind == "line":
            dp, dr = se3.pose_error(P[0], P[1])
            pose_at = lambda u: se3.interpolate(P[0], P[1], u)  # noqa: E731
        else:
            seg_d = [se3.pose_error(a, b) for a, b in zip(P, P[1:])]
            dp, dr = sum(d for d, _ in seg_d), sum(r for _, r in seg_d)
            n = len(P) - 1
            # progress by length, each piece weighted by the cap it uses up, so the
            # timing's speed holds everywhere (the samples need not be evenly spaced)
            w = np.array([max(d / caps.v, r / caps.w) if caps else d + r for d, r in seg_d]) + 1e-12
            cum = np.r_[0.0, np.cumsum(w)] / w.sum()

            def pose_at(u):
                u = min(max(u, 0.0), 1.0)
                i = min(int(np.searchsorted(cum, u, side="right")) - 1, n - 1)
                return se3.interpolate(P[i], P[i + 1], (u - cum[i]) / (cum[i + 1] - cum[i]))
        t, u = TM.unit_timing(max(dp, 1e-9), dr, caps, dt)
        Q = walk(arm, [pose_at(x) for x in u], s.q_start, max_step=0.2)
        if Q is None:
            raise RuntimeError(f"{s.name}: the timed line cannot be followed")
        Q[-1] = s.q_goal
        s.t, s.Q = t, np.array(Q)
    else:
        Q = s.Q
        if Q is None:
            raise RuntimeError(f"{s.name}: no free path")
        jv, ja = float(motion["joint"]["v"]), float(motion["joint"]["a"])
        from .rrt import split_at_corners
        # rest to rest between sharp corners; a smooth piece: a nominal unit-speed
        # parameterisation, then one uniform scale
        ts, qs = [np.array([0.0])], [Q[:1]]
        for piece in split_at_corners(Q):
            d = np.r_[0.0, np.cumsum(np.abs(np.diff(piece, axis=0)).max(axis=1))]
            if d[-1] < 1e-9:
                continue
            T0 = TM.trapezoid(d[-1], jv, ja)
            tt = np.linspace(0.0, T0, max(2, math.ceil(T0 / dt)) + 1)
            sd = TM.trapezoid_s(d[-1], jv, ja, tt)
            Qt = np.column_stack([np.interp(sd, d, piece[:, j]) for j in range(piece.shape[1])])
            tt = tt * TM.joint_retime(tt, Qt, jv, ja)
            ts.append(ts[-1][-1] + tt[1:])
            qs.append(Qt[1:])
        s.t, s.Q = np.concatenate(ts), np.vstack(qs)
    # a carried object keeps its own caps (D14)
    if s.carried is not None and "carry" in motion["caps"] and len(s.t) > 2:
        cc = TM.caps_from(motion["caps"]["carry"])
        for _ in range(8):
            F = [arm.fk(q) for q in s.Q]
            pts = np.array([_carried_point(arm, q, s.carried)[:3] for q in s.Q])
            k = TM.point_kinematics(s.t, pts)
            k.update(TM.angular_kinematics(s.t, np.array([T[:3, :3] for T in F])))
            f = max(1.0, k["v"] / cc.v, math.sqrt(k["a"] / cc.a), math.sqrt(k["a_down"] / cc.a_down),
                    k["w"] / cc.w, math.sqrt(k["alpha"] / cc.alpha))
            if f <= 1.0 + 1e-6:
                break
            s.t = s.t * f * 1.02


# ─────────────────────────────────────────────────────────────────────────────
# Carry orientation: a free move keeps a carried object the way it was picked up
# ─────────────────────────────────────────────────────────────────────────────
_DOWN = np.array([0.0, 0.0, -1.0])


def carry_frame(arm, s: Segment, max_tilt_deg: float = 15.0) -> Optional[tuple]:
    """For a free segment carrying an object: ``(g_nom, c_dir, cos_max)`` in the holding
    link's frame: gravity at the start of the carry, the direction to the object's
    centre, and the cosine of the largest tilt allowed. None otherwise.

    The object rests on the gripper (the tray on the plate, against the push
    plate, between the jaws) and nothing holds it from the open side: a free
    move must not tilt gravity more than *max_tilt_deg* away from where it was
    when the carry began, and gravity must never pull the object outward
    (``g . c_dir <= 0``). The holding link is the arm's tool frame (``arm.fk``)."""
    if s.kind != "free" or s.carried is None:
        return None
    name, link, solids, T = s.carried
    c = np.mean([(np.asarray(T) @ x.T)[:3, 3] for x in solids], axis=0)
    g_nom = arm.fk(s.q_start)[:3, :3].T @ _DOWN
    return g_nom, c / np.linalg.norm(c), math.cos(math.radians(max_tilt_deg))


def carry_sampler(arm, s: Segment, frame, sign_joints=(2, 4), margin: float = 0.3) -> Callable:
    """Configurations that keep a carried object as it was picked up: a random tool
    position around the start and goal, the start orientation turned about the
    vertical (gravity in the tool frame unchanged) and tilted within half the
    allowed cone, on the start's arm branch (IK solution nearest a random blend of
    start and goal)."""
    from rsf.core import se3 as _se3
    T0, T1 = arm.fk(s.q_start), arm.fk(s.q_goal)
    lo = np.minimum(T0[:3, 3], T1[:3, 3]) - margin
    hi = np.maximum(T0[:3, 3], T1[:3, 3]) + margin
    branch = tuple(int(np.sign(s.q_start[j]) or 1) for j in sign_joints)
    cos_max = frame[2]
    half = 0.5 * math.acos(max(-1.0, min(1.0, cos_max)))

    def sample(rng):
        R = _se3.rot_z(rng.uniform(-math.pi, math.pi)) @ T0[:3, :3]
        ax = rng.normal(size=2)
        ax = np.array([ax[0], ax[1], 0.0]) / max(np.linalg.norm(ax), 1e-12)
        R = _se3.axis_angle(ax, rng.uniform(0.0, half)) @ R
        T = _se3.make_T(R, rng.uniform(lo, hi))
        sols = [q for q in arm.ik(T) if tuple(int(np.sign(q[j]) or 1) for j in sign_joints) == branch]
        if not sols:
            return None
        ref = s.q_start + (s.q_goal - s.q_start) * rng.random()
        return min(sols, key=lambda q: float(np.abs(np.asarray(q) - ref).max()))

    return sample


def carry_violation(arm, q, frame) -> Optional[str]:
    """Why *q* breaks the carry orientation (None if it does not)."""
    g_nom, c_dir, cos_max = frame
    g = arm.fk(q)[:3, :3].T @ _DOWN
    if float(g @ g_nom) < cos_max - 1e-9:
        return f"carried object tilted {math.degrees(math.acos(min(1.0, float(g @ g_nom)))):.0f} deg " \
               f"from how it was picked (max {math.degrees(math.acos(cos_max)):.0f})"
    if float(g @ c_dir) > 1e-9:
        return "gravity pulls the carried object outward"
    return None


def _load(checker, s: Segment) -> None:
    """The checker set to segment *s*: its obstacles, attachments and rules."""
    checker.set_obstacles(s.scene.solids)
    for n in list(checker.attached):
        checker.detach(n)
    for name, link, solids, T in s.scene.attached:
        checker.attach(name, link, solids, T)
    checker.policy = s.policy
    checker._rule_cache.clear()


def plan_free_cpu(arm, checker, s: Segment, *, step: float = 0.01, seed: int = 0,
                  max_iter: int = 4000, pad: float = 0.001, max_tilt_deg: float = 15.0) -> dict:
    """CPU fallback for a free segment: RRT-Connect + shortcut + smoothing under the
    gate's rules, every margin *pad* larger (the gate samples the timed path at
    other points than the planner's edge checks; a corner may dip between them).

    Sets ``s.Q`` on success; returns ``{success, seconds, checks, iterations}``."""
    import time as _t
    from dataclasses import replace
    from . import rrt
    _load(checker, s)
    if pad:
        pol = s.policy
        checker.policy = ContactPolicy(pol.default_margin + pad,
                                       [r if r.allow else replace(r, margin=r.margin + pad) for r in pol.rules])
        checker._rule_cache.clear()
    n = {"checks": 0}

    frame = carry_frame(arm, s, max_tilt_deg)

    def valid(q):
        n["checks"] += 1
        if np.any(q < arm.lower) or np.any(q > arm.upper):
            return False
        if frame is not None and carry_violation(arm, q, frame) is not None:
            return False
        d = dict(zip(arm.joint_names, q))
        d.update(s.joints)
        r = checker.check(d)
        if r.ok:
            return True
        # near the ends (taught points) the gate's own margin is enough
        near = min(float(np.abs(q - s.q_start).max()), float(np.abs(q - s.q_goal).max())) < 0.05
        return bool(near and pad and not r.self_violations
                    and all(v.distance >= v.margin - pad - 1e-9 for v in r.violations))

    t0, stats = _t.time(), {}
    path = rrt.rrt_connect(valid, s.q_start, s.q_goal, arm.lower, arm.upper, step=step, seed=seed,
                           max_iter=max_iter, stats=stats,
                           extend=0.15 if frame is not None else 0.3,
                           sampler=carry_sampler(arm, s, frame) if frame is not None else None)
    if path is not None:
        path = rrt.smooth(valid, rrt.shortcut(valid, path, step=step, seed=seed), step=step)
        s.Q = path
    return {"success": path is not None, "seconds": round(_t.time() - t0, 2), "checks": n["checks"],
            "via": 0 if path is None else len(path) - 2, **stats}


def plan_free_gated(arm, checker, s: Segment, motion: dict, *, tries: int = 5, **kw) -> dict:
    """:func:`plan_free_cpu`, then timed and gated as :func:`assemble` will; on a gate
    problem (e.g. a corner dipping below the margin between samples) planned again
    with the next seed. Leaves the untimed path in ``s.Q``."""
    tilt = float(motion.get("carry_max_tilt_deg", 15.0))
    r, why = {"success": False}, []
    for k in range(tries):
        r = plan_free_cpu(arm, checker, s, seed=k, max_tilt_deg=tilt, **kw)
        if not r["success"]:
            continue
        path = s.Q.copy()
        time_segment(arm, s, motion)
        why = check_segment(arm, checker, s, max_tilt_deg=tilt)
        s.Q, s.t = path, None
        if not why:
            r["seed"] = k
            return r
    r.update(success=False, gate=why[:1])
    return r


def check_segment(arm, checker, s: Segment, step: float = 0.01, max_tilt_deg: float = 15.0) -> list:
    """Problems of *s* on the CPU: every joint sample (densified to *step* rad) against its rules."""
    if s.Q is None or s.kind == "hold" or s.scene is None:
        return []
    _load(checker, s)
    Q = s.Q
    dense = [Q[0]]
    for a, b in zip(Q[:-1], Q[1:]):
        k = max(1, math.ceil(float(np.abs(b - a).max()) / step))
        dense += [a + (b - a) * i / k for i in range(1, k + 1)]
    out = []
    for i, q in enumerate(dense):
        d = dict(zip(arm.joint_names, q))
        d.update(s.joints)
        r = checker.check(d)
        if not r.ok:
            v = (r.violations or r.self_violations)[0]
            out.append(f"{s.name}: sample {i}/{len(dense) - 1}: {v.link} vs {v.obstacle} "
                       f"{v.distance * 1e3:.1f} mm (needs {v.margin * 1e3:.1f})")
            break
    if np.any(Q < arm.lower - 1e-9) or np.any(Q > arm.upper + 1e-9):
        out.append(f"{s.name}: outside the operating joint limits")
    frame = carry_frame(arm, s, max_tilt_deg)
    if frame is not None:
        for i, q in enumerate(dense):
            why = carry_violation(arm, q, frame)
            if why:
                out.append(f"{s.name}: sample {i}/{len(dense) - 1}: {why}")
                break
    return out


@dataclass
class Trajectory:
    joint_names: list
    t: np.ndarray
    q: np.ndarray
    segments: list = field(default_factory=list)   #: {name, kind, t0, t1, caps, event}
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        qd = TM.velocities(self.t, self.q)
        return {"schema": "rsf.trajectory/1", "joint_names": self.joint_names, "t": self.t.tolist(),
                "q": np.round(self.q, 7).tolist(), "qd": np.round(qd, 6).tolist(),
                "segments": self.segments, "meta": self.meta}


def assemble(world, arm, checker, segs: list) -> tuple:
    """Time every segment, gate it on the CPU, join them: ``(Trajectory, problems)``."""
    mo = _motion(world.process)
    problems = []
    for s in segs:
        try:
            time_segment(arm, s, mo)
        except RuntimeError as e:
            problems.append(str(e))
            continue
        problems += check_segment(arm, checker, s, max_tilt_deg=float(mo.get("carry_max_tilt_deg", 15.0)))
    jaw_names = sorted({k for s in segs for k in (s.joints or {})} | {k for s in segs for k in (s.joints_end or {})})
    names = list(arm.joint_names) + jaw_names
    T, Qs, out_segs, t0 = [], [], [], 0.0
    q_arm = None
    jaw = {k: None for k in jaw_names}
    for s in segs:
        if s.t is None:
            continue
        if s.kind == "hold":
            if q_arm is None:
                continue
            arm_q = np.repeat(q_arm[None, :], len(s.t), axis=0)
            J = (s.J if s.J is not None else np.array([[jaw[k] for k in jaw_names]] * len(s.t)))
            jn = sorted((s.joints_end or s.joints).keys())
            Jfull = np.array([[J[i][jn.index(k)] if k in jn else jaw[k] for k in jaw_names] for i in range(len(s.t))])
        else:
            arm_q = s.Q
            Jfull = np.array([[s.joints.get(k, jaw[k]) for k in jaw_names]] * len(s.t))
        start = 1 if T else 0
        T += list(t0 + s.t[start:])
        Qs += [np.r_[a, j] for a, j in zip(arm_q[start:], Jfull[start:])]
        out_segs.append({"name": s.name, "kind": s.kind, "t0": round(t0, 4), "t1": round(t0 + float(s.t[-1]), 4),
                         "caps": s.caps if s.kind != "hold" else None, "event": s.event})
        t0 += float(s.t[-1])
        q_arm = arm_q[-1]
        for k, v in zip(jaw_names, Jfull[-1]):
            jaw[k] = v
    Q = np.array([[0.0 if v is None else v for v in row] for row in Qs])
    return Trajectory(names, np.array(T), Q, out_segs, {"layout": world.name}), problems

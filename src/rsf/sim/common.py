"""What every simulator backend shares (P5): parameters, the execution report the
judge reads, and the reference a joint servo follows.

* :class:`SimParams`: the physical model. Most fields are physics, not a
  simulator's (friction, gains, jaw force, timestep); the solver fields
  (``impratio``, ``noslip_iterations``, ``solref``, ``solimp``) are MuJoCo's,
  a backend ignores the ones it has no counterpart for.
* :class:`ExecReport`: what a run did, sampled (``rsf.sim.judge`` reads it).
* :class:`Reference`: the trajectory as a servo setpoint at any time
  (linear in ``q`` and ``qd`` between samples; held before and after).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Joint servo gains per arm joint (N m / rad, N m s / rad), heavy joints first.
ARM_KP = (5.0e4, 5.0e4, 3.0e4, 1.0e4, 1.0e4, 5.0e3)
ARM_KV = (6.0e2, 6.0e2, 3.0e2, 5.0e1, 5.0e1, 2.5e1)


@dataclass
class SimParams:
    timestep: float = 0.001
    friction: dict = field(default_factory=lambda: {"robot": 0.4, "object": 0.3, "static": 0.5})
    arm_kp: tuple = ARM_KP
    arm_kv: tuple = ARM_KV
    arm_ki_per_s: float = 20.0       #: integral gain = kp x this (a load leaves no steady error)
    armature: float = 0.05
    jaw_kp: float = 1.0e5            #: N/m: the EGU50 is screw driven, stiff (1 mm at its 100 N)
    jaw_armature: float = 0.5
    jaw_force: float = 100.0         #: N per jaw (the URDF's effort)
    impratio: float = 10.0           #: elliptic cones, stiff friction (little creep)
    noslip_iterations: int = 10      #: removes the soft-contact creep of objects held by friction
    solref: tuple = (0.002, 1.0)     #: contact time constant (>= 2 timesteps) and damping ratio
    solimp: tuple = (0.95, 0.99, 0.001)
    sample_dt: float = 0.02
    contact_dt: float = 0.005
    contact_tol: float = 0.0         #: penetration (m) beyond which a contact is reported
    settle: float = 0.5              #: s at the start, trajectory held at its first sample


@dataclass
class ExecReport:
    """What a run did. Arrays are per sample (``t``)."""
    t: np.ndarray
    flange: np.ndarray               #: (n, 4, 4) actual
    flange_planned: np.ndarray       #: (n, 4, 4)
    q: np.ndarray                    #: (n, nq_robot) actual joints, trajectory order
    objects: dict                    #: name -> (n, 4, 4) poses
    contacts: list                   #: {t, link, other, depth}
    segments: list                   #: from the trajectory
    meta: dict = field(default_factory=dict)

    def at(self, t: float) -> int:
        return int(min(len(self.t) - 1, np.searchsorted(self.t, t)))



class Reference:
    """``q_ref, qd_ref`` of a ``rsf.trajectory/1`` dict at time *t* (calls in time order)."""

    def __init__(self, traj: dict):
        self.names = list(traj["joint_names"])
        self.T = np.asarray(traj["t"], float)
        self.Q = np.asarray(traj["q"], float)
        qd = traj.get("qd")
        self.QD = np.asarray(qd if qd else np.gradient(self.Q, self.T, axis=0), float)
        self._i = 0
        self._zero = np.zeros(self.Q.shape[1])

    def __call__(self, t: float):
        T, Q, QD = self.T, self.Q, self.QD
        tc = min(max(t, T[0]), T[-1])
        if tc < T[self._i]:
            self._i = 0
        while self._i < len(T) - 2 and T[self._i + 1] < tc:
            self._i += 1
        i = self._i
        u = (tc - T[i]) / max(T[i + 1] - T[i], 1e-12)
        q = Q[i] + (Q[i + 1] - Q[i]) * u
        qd = QD[i] + (QD[i + 1] - QD[i]) * u if 0 <= t <= T[-1] else self._zero
        return q, qd


def T_of(pos, mat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


@dataclass
class View:
    """A camera a backend renders (``cell.add_view`` / ``cell.render``): from *eye*
    toward *target* (world, m), *up* the world z, *fov_deg* vertical, *width* x *height* px."""
    name: str
    eye: tuple
    target: tuple
    fov_deg: float = 45.0
    width: int = 960
    height: int = 540

    @classmethod
    def from_spec(cls, spec, world=None) -> "View":
        """A view from a dict (the fields above), or a preset name (``overview``)."""
        if isinstance(spec, str):
            if spec != "overview":
                raise ValueError(f"unknown view preset {spec!r} (overview, or a dict)")
            return overview(world)
        spec = dict(spec)
        if "name" not in spec or "eye" not in spec or "target" not in spec:
            raise ValueError("a view needs name, eye and target")
        return cls(**{**spec, "eye": tuple(map(float, spec["eye"])), "target": tuple(map(float, spec["target"]))})


def overview(world, name: str = "overview", direction=(1.0, -1.0, 0.6), **kw) -> View:
    """The whole cell: the robot base and every feature in view, seen from *direction*."""
    pts = [np.asarray(world.robot.base, float)[:3, 3]] + [np.asarray(f.T, float)[:3, 3] for f in world.features()]
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    hi[2] = max(hi[2], lo[2] + 1.0)                     # the arm reaches above the features
    c = 0.5 * (lo + hi)
    r = 0.5 * float(np.linalg.norm(hi - lo)) + 0.15
    v = View(name, (0.0, 0.0, 0.0), tuple(float(x) for x in c), **kw)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    dist = r / math.sin(math.radians(0.5 * v.fov_deg * min(1.0, v.width / v.height)))
    v.eye = tuple(float(x) for x in c + dist * d)
    return v


def view_angles(eye, target) -> tuple:
    """``(yaw, pitch)`` in degrees of the direction eye -> target: yaw about the world z
    from +x (counter-clockwise), pitch up from the horizontal."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("eye and target coincide")
    d /= n
    return math.degrees(math.atan2(d[1], d[0])), math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))


def view_forward(yaw: float, pitch: float) -> np.ndarray:
    """The unit direction of *yaw*, *pitch* (degrees, :func:`view_angles`)."""
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])

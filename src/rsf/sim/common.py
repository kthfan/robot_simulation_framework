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

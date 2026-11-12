"""The sim/real seam (D9, D30): ROS2 topics and the payloads on them.

Isaac Sim's internal rclpy (Python 3.11) ships ``sensor_msgs`` and ``std_msgs``
but neither ``trajectory_msgs``, ``control_msgs`` nor ``moveit_msgs``. So the
seam is:

* ``/joint_states`` (``sensor_msgs/JointState``): the cell's joint positions,
  arm and jaws, from the sim or the real robot;
* ``/cell/trajectory`` (``std_msgs/String``): a JSON timed joint trajectory
  for the cell to execute, positions and velocities at times from its start;
* ``/cell/status`` (``std_msgs/String``): JSON reports from the cell, keyed by
  the trajectory id.

Both repos encode and decode through this module, so the payloads cannot drift.
Pure Python + numpy: no ROS here.
"""
from __future__ import annotations

import json
import math

import numpy as np

#: ROS_DOMAIN_ID of the cell (D30).
DOMAIN_ID = 37

ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 7))
#: The jaws, in the stage's DOF order. Both take the same command.
JAW_JOINTS = ("right_gripper_joint", "left_gripper_joint")
JOINTS = ARM_JOINTS + JAW_JOINTS

TOPIC_JOINT_STATES = "/joint_states"
TOPIC_TRAJECTORY = "/cell/trajectory"
TOPIC_STATUS = "/cell/status"

VERSION = 1
#: ``ready``: idle, no trajectory yet; ``accepted``: executing; ``done``:
#: finished, report attached; ``rejected``: not executed, reason attached.
STATES = ("ready", "accepted", "done", "rejected")


class ProtocolError(ValueError):
    """A payload that does not follow this module's format."""


def _rows(a) -> list:
    return [[float(v) for v in row] for row in np.asarray(a, dtype=float)]


def encode_trajectory(tid: str, t, q, qd=None, joint_names=JOINTS, meta=None) -> str:
    """JSON for a timed joint trajectory; *q*, *qd* are ``(len(t), len(joint_names))``."""
    q = np.asarray(q, dtype=float)
    return json.dumps({
        "version": VERSION,
        "id": tid,
        "joint_names": list(joint_names),
        "t": [float(v) for v in t],
        "q": _rows(q),
        "qd": _rows(np.zeros_like(q) if qd is None else qd),
        "meta": meta or {},
    })


def decode_trajectory(text: str) -> dict:
    """The trajectory as numpy arrays, after checking it is executable."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if d.get("version") != VERSION:
        raise ProtocolError(f"version {d.get('version')!r}, expected {VERSION}")
    tid = d.get("id")
    if not isinstance(tid, str) or not tid:
        raise ProtocolError("missing id")
    names = d.get("joint_names")
    if not names or len(set(names)) != len(names) or not set(names) <= set(JOINTS):
        raise ProtocolError(f"joint_names {names!r}")
    try:
        t = np.asarray(d["t"], dtype=float)
        q = np.asarray(d["q"], dtype=float)
        qd = np.asarray(d["qd"], dtype=float)
    except (KeyError, ValueError, TypeError) as e:
        raise ProtocolError(f"arrays: {e}") from e
    n = len(t)
    if t.ndim != 1 or n < 2 or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ProtocolError("t must start at 0 and strictly increase")
    for name, a in (("q", q), ("qd", qd)):
        if a.shape != (n, len(names)):
            raise ProtocolError(f"{name} shape {a.shape}, expected {(n, len(names))}")
        if not np.all(np.isfinite(a)):
            raise ProtocolError(f"{name} not finite")
    if not np.all(np.isfinite(t)):
        raise ProtocolError("t not finite")
    return {"id": tid, "joint_names": list(names), "t": t, "q": q, "qd": qd,
            "meta": d.get("meta") or {}}


def sample(traj: dict, t: float):
    """Position and velocity at time *t*, linear between points.

    Before the start: the first point. After the end: the last position, at rest.
    """
    ts, q, qd = traj["t"], traj["q"], traj["qd"]
    if t >= ts[-1]:
        return q[-1].copy(), np.zeros_like(qd[-1])
    if t <= ts[0]:
        return q[0].copy(), qd[0].copy()
    i = int(np.searchsorted(ts, t, side="right")) - 1
    w = (t - ts[i]) / (ts[i + 1] - ts[i])
    return q[i] + w * (q[i + 1] - q[i]), qd[i] + w * (qd[i + 1] - qd[i])


def encode_status(tid: str, state: str, **fields) -> str:
    if state not in STATES:
        raise ProtocolError(f"state {state!r}")
    return json.dumps({"version": VERSION, "id": tid, "state": state, **fields},
                      default=_jsonable)


def decode_status(text: str) -> dict:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if d.get("version") != VERSION or d.get("state") not in STATES:
        raise ProtocolError(f"status {d.get('version')!r}/{d.get('state')!r}")
    return d


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")

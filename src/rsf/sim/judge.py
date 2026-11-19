"""Judging a simulated run, transfer by transfer (D30 pass criteria). Backend independent.

Reads an :class:`rsf.sim.mujoco_cell.ExecReport` (or any report with the same
fields) and the plan it executed:

1. **placement**: the object ends within ``SEAT_MM`` / ``SEAT_DEG`` of where the
   plan sets it down;
2. **slip**: while carried, the object moves at most ``SLIP_MM`` relative to
   the flange (from where it sat when the carry began);
3. **contacts**: the robot touches nothing but the object it is transferring
   (during that transfer) and, with its base links, the mount it stands on;
4. **tracking**: the flange within ``TRACK_MM`` of the plan where the plan is
   slow and close (segments with the ``in_jig`` caps);
5. **caps**: the carried object's planned motion within the carry caps (D14).

A *transfer* is a pick sequence followed by a place sequence in the trajectory
(``pick:<from>/...``, ``place:<to>/...``). Where the carry begins and ends is
named by phases (``carry_from``: the end of a pick segment, ``carry_to``: the
start of a place segment); the TM tray cell's are the defaults.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rsf.planning import timing as TM

SEAT_MM = 2.0
SEAT_DEG = 1.0
SLIP_MM = 5.0
TRACK_MM = 1.0
TRACK_CAPS = ("in_jig",)
#: Robot links that stand on their mount (contacts with ``robot_seat`` solids allowed).
BASE_LINKS = ("base", "link_0", "link_1")

_SEQ = re.compile(r"^(?:free\d+:)?((?:pick|place):.+?)/([^/:]+)(?::act)?$")


def pose_error(a, b) -> tuple:
    """``(mm, deg)`` between two 4x4 poses."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = max(-1.0, min(1.0, (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1e3, math.degrees(math.acos(c))


@dataclass
class Transfer:
    src: str
    dst: str
    obj: str                      #: the simulated body moved
    t0: float
    t1: float
    carry: tuple                  #: (t_start, t_end) carried on the gripper
    seat_T: np.ndarray            #: where the plan sets it down
    result: dict = field(default_factory=dict)


def sequences(segments: list) -> list:
    """``[(sequence, phase, segment)]`` of a trajectory's segments that belong to a sequence."""
    out = []
    for s in segments:
        m = _SEQ.match(s["name"])
        if m:
            out.append((m.group(1), m.group(2), s))
    return out


def planned_object_pose(world, seq, obj_template) -> Optional[np.ndarray]:
    """Where the plan leaves the object at the end of *seq* (an ``rsf.planning.teach.Sequence``):
    its pose in the last waypoint's scene."""
    want = {f"{obj_template.name}/{s.name}": s for s in obj_template.solids}
    for s in seq.waypoints[-1].scene.solids:
        if s.name in want:
            return s.T @ np.linalg.inv(want[s.name].T)
    return None


def transfers(world, workflow, segments: list, occupancy: dict, *,
              carry_from: str = "lift:act", carry_to: str = "place_down") -> list:
    """The run's transfers, with the body each one moves (tracked from *occupancy*)."""
    from rsf.model import registry
    from rsf.model.workflow import expand
    seqs = {}
    for task in expand(workflow, world).tasks:
        if task.skill in registry.names("skill"):
            for sq in registry.get("skill", task.skill)(world, task.args):
                seqs[sq.name] = sq
    occ = dict(occupancy)
    rows = sequences(segments)
    out, i = [], 0
    while i < len(rows):
        name, _, _ = rows[i]
        if not name.startswith("pick:"):
            i += 1
            continue
        j = i
        while j < len(rows) and rows[j][0] == name:
            j += 1
        if j >= len(rows) or not rows[j][0].startswith("place:"):
            break
        k = j
        while k < len(rows) and rows[k][0] == rows[j][0]:
            k += 1
        # a taught variant ("pick:<slot>#<configuration>") is the same slot
        src, dst = name[len("pick:"):].split("#")[0], rows[j][0][len("place:"):].split("#")[0]
        pick, place = rows[i:j], rows[j:k]
        c0 = next((s["t1"] for _, ph, s in pick if s["name"].endswith(carry_from)), pick[-1][2]["t1"])
        c1 = next((s["t0"] for _, ph, s in place if ph == carry_to and not s["name"].endswith(":act")),
                  place[-1][2]["t0"])
        obj = occ.pop(src, None)
        if obj is not None:
            occ[dst] = obj
        tmpl = world.object_at(dst)
        seat = planned_object_pose(world, seqs[rows[j][0].split("#")[0]], tmpl) if tmpl is not None else None
        out.append(Transfer(src, dst, obj, pick[0][2]["t0"], place[-1][2]["t1"], (c0, c1),
                            tmpl.T if seat is None else seat))
        i = k
    return out


def _window(rep, t0, t1):
    return np.flatnonzero((rep.t >= t0 - 1e-9) & (rep.t <= t1 + 1e-9))


def _planned_flange(world, traj):
    """``(t, flange poses)`` of a trajectory at its own samples (for the caps)."""
    from rsf.kinematics import load_arm
    arm = load_arm(world)
    T = np.asarray(traj["t"], float)
    Q = np.asarray(traj["q"], float)
    idx = [traj["joint_names"].index(n) for n in arm.joint_names]
    return T, Q[:, idx], arm


def judge(world, workflow, rep, occupancy: dict, *, carry_caps=None, traj: Optional[dict] = None, **kw) -> dict:
    """Every transfer's verdict and the run's: ``{ok, transfers: [...], other_contacts}``.

    With *traj* (the executed ``rsf.trajectory/1``), the carry caps are judged
    on the plan's own samples, as the planner caps them; otherwise on the
    report's planned flange samples (coarser: a corner between them reads
    differently with every alignment)."""
    planned = _planned_flange(world, traj) if traj is not None else None
    seat_solids = {s.name for s in world.solids(objects=False) if "robot_seat" in s.tags}
    if carry_caps is None:
        mo = (world.process.get("motion") or {}).get("caps", {})
        carry_caps = TM.caps_from(mo["carry"]) if "carry" in mo else None
    trs = [t for t in transfers(world, workflow, rep.segments, occupancy, **kw)
           if t.t1 <= rep.t[-1] + 1e-9]                  # only transfers the run completed
    used = set()
    for tr in trs:
        r = {}
        # 1. placement
        if tr.obj in rep.objects:
            r["seat_mm"], r["seat_deg"] = pose_error(rep.objects[tr.obj][rep.at(tr.t1)], tr.seat_T)
        else:
            r["seat_mm"], r["seat_deg"] = math.inf, math.inf
        # 2. slip relative to the flange while carried
        w = _window(rep, *tr.carry)
        slip, rel_ref = math.inf, None
        if tr.obj in rep.objects and len(w):
            rel = np.array([np.linalg.inv(rep.flange[i]) @ rep.objects[tr.obj][i] for i in w])
            rel_ref = rel[0]
            slip = float(np.linalg.norm(rel[:, :3, 3] - rel_ref[:3, 3], axis=1).max()) * 1e3
        r["slip_mm"] = slip
        # 3. contacts during the transfer
        bad = []
        for n, c in enumerate(rep.contacts):
            if tr.t0 - 1e-9 <= c["t"] <= tr.t1 + 1e-9:
                used.add(n)
                if c["other"] == tr.obj or (c["link"] in BASE_LINKS and c["other"] in seat_solids):
                    continue
                bad.append(c)
        r["contacts"] = _summary(bad)
        # 4. tracking where the plan is slow and close
        trk = 0.0
        for s in rep.segments:
            if s.get("caps") in TRACK_CAPS and s["t0"] >= tr.t0 - 1e-9 and s["t1"] <= tr.t1 + 1e-9:
                for i in _window(rep, s["t0"], s["t1"]):
                    trk = max(trk, pose_error(rep.flange[i], rep.flange_planned[i])[0])
        r["track_mm"] = trk
        # 5. the carried object's planned motion within the carry caps
        r["caps_ok"], r["carry_kin"] = True, None
        if carry_caps is not None and rel_ref is not None and len(w) > 3:
            if planned is not None:
                Tp, Qp, arm = planned
                wp = np.flatnonzero((Tp >= tr.carry[0] - 1e-9) & (Tp <= tr.carry[1] + 1e-9))
                tk = Tp[wp]
                P = np.array([arm.fk(q) @ rel_ref for q in Qp[wp]])
            else:
                tk = rep.t[w]
                P = np.array([rep.flange_planned[i] @ rel_ref for i in w])
            k = TM.point_kinematics(tk, P[:, :3, 3])
            k.update(TM.angular_kinematics(tk, P[:, :3, :3]))
            r["carry_kin"] = {a: round(b, 4) for a, b in k.items()}
            r["caps_ok"] = (k["v"] <= carry_caps.v * 1.05 and k["a"] <= carry_caps.a * 1.05
                            and k["w"] <= carry_caps.w * 1.05 and k["alpha"] <= carry_caps.alpha * 1.05
                            and k["a_down"] <= carry_caps.a_down * 1.05)
        r["checks"] = {"placement": r["seat_mm"] <= SEAT_MM and r["seat_deg"] <= SEAT_DEG,
                       "slip": r["slip_mm"] <= SLIP_MM, "contacts": not bad,
                       "tracking": r["track_mm"] <= TRACK_MM, "caps": bool(r["caps_ok"])}
        r["ok"] = all(r["checks"].values())
        tr.result = r
    # contacts outside every transfer (free moves between them, to and from home)
    t_last = max((t.t1 for t in trs), default=-math.inf)
    other = [c for n, c in enumerate(rep.contacts) if n not in used and c["t"] <= t_last + 1e-9
             and not (c["link"] in BASE_LINKS and c["other"] in seat_solids)]
    return {"ok": all(t.result["ok"] for t in trs) and not other,
            "transfers": [{"src": t.src, "dst": t.dst, "object": t.obj, "t0": round(t.t0, 3),
                           "t1": round(t.t1, 3), "carry": [round(x, 3) for x in t.carry],
                           **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.result.items()}}
                          for t in trs],
            "other_contacts": _summary(other)}


def _summary(contacts: list) -> list:
    """Contacts grouped by (link, other): count, first/last time, deepest penetration (mm)."""
    g = {}
    for c in contacts:
        k = (c["link"], c["other"])
        e = g.setdefault(k, {"link": k[0], "other": k[1], "n": 0, "t_first": c["t"], "t_last": c["t"], "depth_mm": 0.0})
        e["n"] += 1
        e["t_last"] = c["t"]
        e["depth_mm"] = round(max(e["depth_mm"], c["depth"] * 1e3), 3)
    return sorted(g.values(), key=lambda e: -e["depth_mm"])

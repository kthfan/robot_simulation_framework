"""Automatic teach points (P3): one feasible, continuous joint configuration per waypoint.

A skill (registered by a cell plugin) turns a task into :class:`Sequence` s of
:class:`Waypoint` s: a tool pose, the scene at that moment (obstacles, attached
objects), the contact rules, extra joint values (jaws), and how the tool
*arrives* from the previous waypoint:

* ``free``: any collision-free joint motion (planned in P4); the branch may change;
* ``line``: a straight tool line (position linear, rotation geodesic);
* ``path``: the given tool poses (e.g. the tilt motion).

The engine:

1. solves every waypoint for all exact, in-limit IK branches and checks each
   against its scene and rules;
2. splits the sequence into *runs* joined by ``free`` arrivals; inside a run it
   walks every passing start branch through the line/path arrivals with one
   continuous IK solution per sample (``rsf.planning.cartesian.walk``),
   collision-checking every sample;
3. keeps, per run, the chain with the largest worst-case slack (clearance beyond
   the required margin), then joint travel; the longest run anchors the others,
   which take the chain nearest to it in joint space. With a *preference*
   (``process.continuity.prefer``) chains in the preferred arm configuration
   come first and every chain is turned (whole turns per joint) to lie nearest
   a reference pose; across many sequences (a slot table, a workflow) the
   preferred configuration is by default the one every sequence can use with
   the largest worst-case slack (:func:`choose_branch`), so the free moves
   between sequences never flip the elbow or the wrist; in a workflow each
   sequence is then turned by whole turns so the run travels least between
   sequences (:func:`align_turns`);
4. reports a :class:`TeachPoint` per waypoint: joints, tool pose, clearance and
   the obstacle it is measured to, the worst slack along the arrival motion,
   manipulability, joint-limit margin and the arm configuration (branch signs).

Nothing here is cell specific; the TM tray cell's pick and place are skills in
its plugin.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from rsf.collision import ContactPolicy
from rsf.core import se3
from rsf.planning.cartesian import line_poses, walk
from rsf.planning.chain import ContinuityRule


@dataclass(frozen=True, eq=False)
class Scene:
    """What the robot must not hit at a waypoint."""
    solids: tuple                    #: world solids
    attached: tuple = ()             #: (name, link, solids in the object frame, T_link_object)


@dataclass(frozen=True, eq=False)
class Waypoint:
    name: str
    T: np.ndarray                    #: tool pose, world frame
    scene: Scene
    policy: ContactPolicy
    joints: dict = field(default_factory=dict)       #: other joints (jaws) while arriving and here
    arrive: str = "free"                            #: free | line | path
    path: tuple = ()                                #: poses for ``path`` (previous -> this, both included)
    along_policy: Optional[ContactPolicy] = None    #: rules along the arrival (default: ``policy``)
    along_scene: Optional[Scene] = None             #: scene along the arrival (default: ``scene``)
    action: str = ""                                #: what is done here (shown in the teach table)
    meta: dict = field(default_factory=dict)


@dataclass
class Sequence:
    name: str
    waypoints: list
    meta: dict = field(default_factory=dict)


@dataclass
class TeachPoint:
    sequence: str
    name: str
    q: list                       #: arm joints, rad
    xyz: list                     #: tool position, m (world)
    quat_wxyz: list               #: tool orientation (world)
    joints: dict                  #: other joints commanded here
    arrive: str
    clearance: Optional[float]    #: distance to ``nearest`` (m), the smallest-slack pair
    margin: Optional[float]       #: what that pair had to keep
    nearest: Optional[list]       #: [robot link, obstacle]
    along_slack: Optional[float]  #: worst slack along the arrival motion (line/path)
    along_samples: int
    manipulability: float
    limit_margin: float           #: smallest distance to a joint limit, rad
    branch: dict                  #: sign of each configuration joint
    action: str = ""


@dataclass
class TeachResult:
    sequence: str
    ok: bool
    points: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"sequence": self.sequence, "ok": self.ok, "problems": self.problems, "meta": self.meta,
                "points": [asdict(p) for p in self.points]}


@dataclass
class _Eval:
    ok: bool
    slack: float
    worst: object


@dataclass
class Explored:
    """Every feasible chain of every run of a sequence: ``options[run] = [(qs, worst, along)]``."""
    seq: Sequence
    runs: list
    options: list
    problems: list


def choose_branch(engine, explored: list) -> Optional[dict]:
    """The arm configuration for a set of sequences: usable by as many of them as
    possible, then with the largest worst-case slack over them. ``{joint: +-1}``."""
    score = {}
    for ex in explored:
        if ex.problems:
            continue
        per = {}
        for opts in ex.options:                        # a sequence has a configuration if every run does
            best = {}
            for qs, w, _ in opts:
                sig = engine.signature(qs)
                if sig is not None:
                    best[sig] = max(best.get(sig, -math.inf), w)
            per = best if not per else {k: min(per[k], v) for k, v in per.items() if k in best}
            if not per:
                break
        for sig, w in per.items():
            n, lo = score.get(sig, (0, math.inf))
            score[sig] = (n + 1, min(lo, w))
    if not score:
        return None
    sig = max(score, key=lambda k: score[k])
    return {engine.arm.joint_names[j]: g for j, g in zip(engine.rule.sign_joints, sig)}


def resolve_prefer(world, engine, explored: list) -> Optional[dict]:
    """``process.continuity.prefer`` made concrete: ``branch: auto`` (default) is
    :func:`choose_branch`; ``q_ref`` defaults to ``process.motion.home``."""
    c = (world.process.get("continuity") or {}).get("prefer", {"branch": "auto"})
    if c in (None, False, "none"):
        return None
    out = dict(c)
    if out.get("branch", "auto") == "auto":
        out["branch"] = choose_branch(engine, explored)
    if out.get("q_ref") is None:
        home = (world.process.get("motion") or {}).get("home")
        out["q_ref"] = None if home is None else [float(v) for v in home]
    return out


class TeachEngine:
    """Teach points for sequences, with one robot and one collision checker."""

    def __init__(self, arm, checker, *, rule: ContinuityRule = ContinuityRule(),
                 line_step: float = 0.005, line_ang_step: float = math.radians(1.0),
                 max_step: float = 0.2, max_slack: float = 0.05, prefer: Optional[dict] = None):
        self.arm, self.checker, self.rule = arm, checker, rule
        #: ``{"branch": {joint: +-1}, "q_ref": [..]}`` or ``None``; see :meth:`select`
        self.prefer = prefer
        self.line_step, self.line_ang_step, self.max_step = line_step, line_ang_step, max_step
        self.max_slack = max_slack       #: slack beyond this counts as "far" when ranking
        self._scene_id = None

    # ── evaluation ─────────────────────────────────────────────────────────
    def _use(self, scene: Scene, policy: ContactPolicy) -> None:
        if self._scene_id != id(scene):
            self.checker.set_obstacles(scene.solids)
            for name in list(self.checker.attached):
                self.checker.detach(name)
            for name, link, solids, T in scene.attached:
                self.checker.attach(name, link, solids, T)
            self._scene_id = id(scene)
        if self.checker.policy is not policy:
            self.checker.policy = policy
            self.checker._rule_cache.clear()

    def _q(self, q6, wp: Waypoint) -> dict:
        d = dict(zip(self.arm.joint_names, np.asarray(q6, float)))
        d.update(wp.joints)
        return d

    def _eval(self, q6, wp: Waypoint, scene: Scene, policy: ContactPolicy) -> _Eval:
        self._use(scene, policy)
        r = self.checker.check(self._q(q6, wp))
        slack = self.max_slack if r.worst is None else min(self.max_slack, r.worst.slack)
        if r.self_violations:
            slack = min(slack, min(v.slack for v in r.self_violations))
        return _Eval(r.ok, slack, r.worst)

    def _arrival_poses(self, prev: Waypoint, wp: Waypoint) -> list:
        if wp.arrive == "path":
            return list(wp.path)
        return line_poses(prev.T, wp.T, step=self.line_step, ang_step=self.line_ang_step)

    def _follow(self, run: list, q0) -> tuple:
        """Walk *run* from *q0* (at ``run[0]``): ``(qs, worst slack, along slacks)`` or ``(None, why)``."""
        qs, worst, along = [np.asarray(q0, float)], math.inf, [None]
        e = self._eval(q0, run[0], run[0].scene, run[0].policy)
        if not e.ok:
            return None, f"{run[0].name}: start in collision"
        worst = e.slack
        for prev, wp in zip(run, run[1:]):
            poses = self._arrival_poses(prev, wp)
            seg = walk(self.arm, poses, qs[-1], max_step=self.max_step, rule=self.rule)
            if seg is None:
                return None, f"{prev.name} -> {wp.name}: no continuous IK ({wp.arrive})"
            scene, pol = wp.along_scene or wp.scene, wp.along_policy or wp.policy
            seg_worst = math.inf
            for k, q in enumerate(seg[1:-1], start=1):
                e = self._eval(q, wp, scene, pol)
                if not e.ok:
                    w = e.worst
                    what = f"{w.link} vs {w.obstacle} {w.distance * 1e3:.1f} mm (needs {w.margin * 1e3:.1f})" if w else "self"
                    return None, f"{prev.name} -> {wp.name}: sample {k}/{len(seg) - 1}: {what}"
                seg_worst = min(seg_worst, e.slack)
            e = self._eval(seg[-1], wp, wp.scene, wp.policy)
            if not e.ok:
                return None, f"{wp.name}: end in collision"
            seg_worst = min(seg_worst, e.slack)
            if wp.arrive == "path" and wp.along_policy is not None:
                # a path's own rules hold to its end (the tilt's clearance includes the carry pose)
                e = self._eval(seg[-1], wp, scene, pol)
                if not e.ok:
                    w = e.worst
                    return None, (f"{prev.name} -> {wp.name}: end: {w.link} vs {w.obstacle} "
                                  f"{w.distance * 1e3:.1f} mm (needs {w.margin * 1e3:.1f})")
                seg_worst = min(seg_worst, e.slack)
            along.append(seg_worst)
            worst = min(worst, seg_worst)
            qs.append(seg[-1])
        return qs, worst, along

    # ── selection ──────────────────────────────────────────────────────────
    def _shift_to(self, qs: list, ref) -> list:
        """*qs* moved by whole turns, joint by joint, so ``qs[0]`` is nearest *ref*, staying in limits."""
        Q = np.array(qs)
        for j in range(Q.shape[1]):
            best, bd = 0, abs(Q[0, j] - ref[j])
            for k in (-2, -1, 1, 2):
                col = Q[:, j] + 2 * math.pi * k
                if np.all(col >= self.arm.lower[j] - 1e-9) and np.all(col <= self.arm.upper[j] + 1e-9):
                    if abs(col[0] - ref[j]) < bd:
                        best, bd = k, abs(col[0] - ref[j])
            Q[:, j] += 2 * math.pi * best
        return list(Q)

    def signature(self, qs) -> Optional[tuple]:
        """The arm configuration of a chain (signs of the continuity joints), ``None`` if mixed."""
        sig = {tuple(int(np.sign(q[j]) or 1) for j in self.rule.sign_joints) for q in qs}
        return next(iter(sig)) if len(sig) == 1 else None

    def teach(self, seq: Sequence, prefer: Optional[dict] = None) -> TeachResult:
        return self.select(self.explore(seq), prefer if prefer is not None else self.prefer)

    def explore(self, seq: Sequence) -> "Explored":
        """Every feasible chain of every run of *seq* (the expensive part)."""
        wps = seq.waypoints
        runs, cur = [], []
        for wp in wps:
            if wp.arrive == "free" and cur:
                runs.append(cur)
                cur = []
            cur.append(wp)
        runs.append(cur)

        options = []      # per run: list of (qs, worst, along)
        problems = []
        for run in runs:
            found, why = [], []
            for q in self.arm.ik(run[0].T, expand_turns=False):
                got = self._follow(run, q)
                if got[0] is None:
                    why.append(got[1])
                else:
                    found.append(got)
            if not found:
                problems.append(f"run {run[0].name}..{run[-1].name}: " + ("; ".join(sorted(set(why))[:3]) or "no IK"))
            options.append(found)
        return Explored(seq, runs, options, problems)

    def select(self, ex: "Explored", prefer: Optional[dict] = None) -> TeachResult:
        """One chain per run of an explored sequence (see the module notes, step 3)."""
        seq, runs, options, problems = ex.seq, ex.runs, ex.options, list(ex.problems)
        wps = seq.waypoints
        if problems:
            return TeachResult(seq.name, False, [], problems, dict(seq.meta))
        want = None
        if prefer and prefer.get("branch"):
            names = self.arm.joint_names
            want = tuple(int(prefer["branch"].get(names[j], 0)) for j in self.rule.sign_joints)
        q_ref = None if not prefer or prefer.get("q_ref") is None else np.asarray(prefer["q_ref"], float)

        def mismatch(qs):
            if want is None:
                return 0
            sig = self.signature(qs)
            return 0 if sig is not None and all(w in (0, g) for w, g in zip(want, sig)) else 1

        travel = lambda qs: float(sum(np.abs(np.diff(np.array(qs), axis=0)).max(axis=1))) if len(qs) > 1 else 0.0  # noqa: E731
        anchor = max(range(len(runs)), key=lambda i: len(runs[i]))
        chosen = [None] * len(runs)
        qs, worst, along = max(options[anchor], key=lambda o: (-mismatch(o[0]), round(o[1], 4), -travel(o[0])))
        if q_ref is not None:
            qs = self._shift_to(qs, q_ref)
        chosen[anchor] = (qs, worst, along)
        for i in list(range(anchor + 1, len(runs))) + list(range(anchor - 1, -1, -1)):
            nb = chosen[i - 1] if i > anchor else chosen[i + 1]
            ref = nb[0][-1] if i > anchor else nb[0][0]
            best = None
            for o_qs, o_w, o_al in options[i]:
                cand = self._shift_to(o_qs, ref) if i > anchor else self._shift_to(o_qs[::-1], ref)[::-1]
                d = float(np.abs((cand[0] if i > anchor else cand[-1]) - ref).max())
                key = (mismatch(cand), d, -o_w)
                if best is None or key < best[0]:
                    best = (key, (cand, o_w, o_al))
            chosen[i] = best[1]

        points = []
        for run, (qs, _, along) in zip(runs, chosen):
            for wp, q, al in zip(run, qs, along):
                e = self._eval(q, wp, wp.scene, wp.policy)
                w = e.worst
                lo, hi = self.arm.lower, self.arm.upper
                T = self.arm.fk(q)
                points.append(TeachPoint(
                    seq.name, wp.name, [round(float(v), 6) for v in q],
                    [round(float(v), 6) for v in T[:3, 3]], [round(float(v), 6) for v in se3.quat_wxyz(T[:3, :3])],
                    {k: round(float(v), 6) for k, v in wp.joints.items()}, wp.arrive,
                    None if w is None else round(w.distance, 5), None if w is None else w.margin,
                    None if w is None else [w.link, w.obstacle],
                    None if al is None else round(al, 5),
                    0 if wp.arrive == "free" or al is None else len(self._arrival_poses(run[run.index(wp) - 1], wp)) - 1,
                    round(self.arm.manipulability(q), 6),
                    round(float(min(np.min(q - lo), np.min(hi - q))), 6),
                    {self.arm.joint_names[j]: int(np.sign(q[j]) or 1) for j in self.rule.sign_joints},
                    wp.action))
        # sanity: the reported pose is the waypoint's
        for p, wp in zip(points, wps):
            dp, dr = se3.pose_error(self.arm.fk(np.array(p.q)), wp.T)
            if dp > 1e-6 or dr > 1e-5:
                problems.append(f"{wp.name}: teach point misses its pose by {dp * 1e3:.3f} mm")
        return TeachResult(seq.name, not problems, points, problems, dict(seq.meta))


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────
def make_engine(world, **kw) -> TeachEngine:
    """An engine for the world's robot: EAIK kinematics, coal collision.

    ``process.continuity`` (``{sign_joints: [...], max_step: rad}``) sets what
    counts as one arm configuration; the default suits 6R arms with an elbow at
    joint 3 and a wrist at joint 5.
    """
    from rsf.collision.coal_backend import CoalChecker
    from rsf.kinematics import load_arm
    c = world.process.get("continuity") or {}
    rule = ContinuityRule(tuple(c.get("sign_joints", (2, 4))), float(c.get("max_step", 1.0)))
    arm = load_arm(world)
    checker = CoalChecker(world.robot.urdf, world.robot.base, ContactPolicy(0.0), cutoff=0.06)
    return TeachEngine(arm, checker, rule=rule, **kw)


def teach_targets(world, skills, select=None, engine: Optional[TeachEngine] = None, progress=None,
                  info: Optional[dict] = None) -> list:
    """Every sequence of every *skills* (names) at every selected feature (``{"at": key}``)."""
    from rsf.model import registry
    engine = engine or make_engine(world)
    explored = []
    for f in world.features(select or {"kind": "slot"}):
        for name in skills:
            for seq in registry.get("skill", name)(world, {"at": f.key}):
                explored.append(engine.explore(seq))
    prefer = resolve_prefer(world, engine, explored)
    if info is not None:
        info["prefer"] = prefer
    out = []
    for ex in explored:
        out.append(engine.select(ex, prefer))
        if progress:
            progress(out[-1])
    return out


def align_turns(engine, seqs: dict, order: list, home=None, max_combos: int = 4096) -> float:
    """Whole turns per joint and sequence so the run travels least between sequences.

    *order* lists sequence names as the run executes them (a sequence may recur);
    the run starts and ends at *home*. The cost, the sum over transitions of
    ``|q_end(prev) - q_start(next)|`` per joint, separates by joint. Per joint:
    every combination of turns of the recurring sequences (few: the shared
    pick/place of a workflow), then a Viterbi pass over the run for the
    sequences that occur once -- exact. A turn is admissible when every point of
    the sequence stays inside the limits. Returns the travel in rad."""
    import itertools
    seq = [n for n in order if seqs[n].ok]
    names = list(dict.fromkeys(seq))
    if not names:
        return 0.0
    lo, hi = engine.arm.lower, engine.arm.upper
    Q = {n: np.array([p.q for p in seqs[n].points], float) for n in names}
    counts = {n: seq.count(n) for n in names}
    total = 0.0
    shift = {n: np.zeros(Q[n].shape[1]) for n in names}
    for j in range(Q[names[0]].shape[1]):
        tw = 2 * math.pi
        opts = {n: [k for k in (-2, -1, 0, 1, 2)
                    if np.all(Q[n][:, j] + tw * k >= lo[j] - 1e-9) and np.all(Q[n][:, j] + tw * k <= hi[j] + 1e-9)]
                or [0] for n in names}
        multi = [n for n in names if counts[n] > 1]
        combos = list(itertools.product(*(opts[n] for n in multi)))
        if len(combos) > max_combos:                       # too many: recurring ones stay nearest zero
            combos = [tuple(min(opts[n], key=abs) for n in multi)]
        best = (math.inf, None)
        for combo in combos:
            fixed = dict(zip(multi, combo))
            # Viterbi over occurrences; state = turn of that occurrence's sequence
            prev = {None: (0.0 if home is None else 0.0, {})}
            last_end = None
            for i, n in enumerate(seq):
                states = [fixed[n]] if n in fixed else opts[n]
                cur = {}
                for k in states:
                    s0 = Q[n][0, j] + tw * k
                    cands = []
                    for pk, (c, asg) in prev.items():
                        if i == 0:
                            e = None if home is None else home[j]
                        else:
                            e = Q[seq[i - 1]][-1, j] + tw * pk
                        cands.append((c + (0.0 if e is None else abs(e - s0)), pk, asg))
                    c, pk, asg = min(cands, key=lambda x: x[0])
                    a2 = dict(asg)
                    a2[n] = k
                    cur[k] = (c, a2)
                prev = cur
            for k, (c, asg) in prev.items():
                if home is not None:
                    c += abs(Q[seq[-1]][-1, j] + tw * k - home[j])
                if c < best[0] - 1e-12:
                    best = (c, asg)
        total += best[0]
        for n in names:
            shift[n][j] = tw * best[1].get(n, 0)
    for n in names:
        for p, q in zip(seqs[n].points, Q[n] + shift[n]):
            p.q = [round(float(v), 6) for v in q]
            p.limit_margin = round(float(min(np.min(q - lo), np.min(hi - q))), 6)
    return total


def teach_workflow(world, workflow, engine: Optional[TeachEngine] = None, progress=None) -> dict:
    """Teach points for every motion a workflow run needs, each taught once.

    Returns ``{"tasks": [...], "sequences": {name: TeachResult}, "end": event}``;
    a task lists the sequences it executes, in order. Skills without a
    registered motion (e.g. ``wait``) have none.
    """
    from rsf.model import registry
    from rsf.model.workflow import expand
    engine = engine or make_engine(world)
    ex = expand(workflow, world)
    explored, tasks = {}, []
    for t in ex.tasks:
        names = []
        if t.skill in registry.names("skill"):
            for seq in registry.get("skill", t.skill)(world, t.args):
                if seq.name not in explored:
                    explored[seq.name] = engine.explore(seq)
                names.append(seq.name)
        tasks.append({"index": t.index, "cycle": t.cycle, "skill": t.skill, "args": t.args, "sequences": names})
    prefer = resolve_prefer(world, engine, list(explored.values()))
    seqs = {name: engine.select(e, prefer) for name, e in explored.items()}
    # A sequence that ends holding an object must hand over to the next one in the same
    # arm configuration: the configuration may change only with the gripper empty. If the
    # next one needs another configuration (a slot the preferred one cannot serve), the
    # earlier sequence is taught again in that configuration, for that task only.
    for t in tasks:
        names = t["sequences"]
        for i in range(len(names) - 1):
            a, b = names[i], names[i + 1]
            ra, rb = seqs.get(a), seqs.get(b)
            if not (ra and rb and ra.ok and rb.ok) or not explored[a].seq.waypoints[-1].scene.attached:
                continue
            want = rb.points[0].branch
            if ra.points[-1].branch == want:
                continue
            var = f"{a}#" + ",".join(f"{k}{v:+d}" for k, v in sorted(want.items()))
            if var not in seqs:
                r = engine.select(explored[a], dict(prefer or {}, branch=dict(want)))
                if r.ok and r.points[-1].branch == want:
                    r.sequence = var
                    for p in r.points:
                        p.sequence = var
                    r.meta = dict(r.meta, variant_of=a, branch=dict(want))
                    seqs[var] = r
            if var in seqs:
                names[i] = var
    order = [n for t in tasks for n in t["sequences"]]
    home = None if not prefer or prefer.get("q_ref") is None else np.asarray(prefer["q_ref"], float)
    align_turns(engine, seqs, order, home)
    for r in seqs.values():
        if progress:
            progress(r)
    return {"tasks": tasks, "sequences": seqs, "end": {"name": ex.end.name, "detail": ex.end.detail},
            "prefer": prefer}


def write_table(results, path_json, path_csv=None, meta: Optional[dict] = None) -> None:
    """Teach results as JSON (everything) and CSV (one row per point, degrees and mm)."""
    import csv
    import json
    from pathlib import Path
    results = list(results)
    doc = {"meta": meta or {}, "ok": all(r.ok for r in results), "sequences": [r.to_dict() for r in results]}
    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    Path(path_json).write_text(json.dumps(doc, indent=1))
    if path_csv:
        with open(path_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "point", "arrive", "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
                        "x_mm", "y_mm", "z_mm", "qw", "qx", "qy", "qz", "jaw", "clearance_mm", "nearest",
                        "along_slack_mm", "manipulability", "limit_margin_deg", "action"])
            for r in results:
                for p in r.points:
                    jaw = next(iter(p.joints.values()), "")
                    w.writerow([r.sequence, p.name, p.arrive] + [round(math.degrees(v), 4) for v in p.q]
                               + [round(v * 1e3, 3) for v in p.xyz] + p.quat_wxyz
                               + [jaw, None if p.clearance is None else round(p.clearance * 1e3, 2),
                                  "" if p.nearest is None else f"{p.nearest[0]} -> {p.nearest[1]}",
                                  None if p.along_slack is None else round(p.along_slack * 1e3, 2),
                                  p.manipulability, round(math.degrees(p.limit_margin), 2), p.action])

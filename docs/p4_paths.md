# P4 — paths: a workflow run as one timed, checked joint trajectory

Date: 2026-11-16. Code: `src/rsf/planning/{plan,timing,rrt}.py`,
`tools/plan_workflow.py`, `tools/kaggle/curobo_free_plan.py`. Output:
`cells/tm_tray/plans/`.

## Division of work

| part | where | how |
|---|---|---|
| teach points (P3) | CPU | EAIK + coal, every sample checked |
| constrained motions: lines in the shelf, the tilt | CPU | continuous IK walk along the timed tool path |
| **free moves** (between teach points, with or without the tray) | **GPU offline (cuRobo, Kaggle for now)** | `plan_cspace`, exact mesh check on, limits inside the margin |
| free moves without a GPU (development, CI) | CPU | RRT-Connect + shortcut + Chaikin smoothing (`--paths cpu`) |
| timing, the gate, the trajectory | CPU | every sample re-checked with the same rules as the teach points |

The GPU result is imported, never trusted: every free path is re-checked on the
CPU (coal, exact meshes, 0.01 rad samples) before it becomes part of a
trajectory. A path that fails the gate is a problem in the report, not a
trajectory.

```bash
# 1. teach the workflow (P3): one configuration for the run, whole turns chosen for the run
PYTHONPATH=src python -m rsf teach cells/tm_tray/layouts/strict.yaml \
    --workflow cells/tm_tray/workflows/inspect_and_sort.yaml --out cells/tm_tray/teach/strict_inspect_and_sort
# 2. the free moves as a cuRobo job (commit it: Kaggle clones the repository)
PYTHONPATH=src python tools/plan_workflow.py job cells/tm_tray/layouts/strict.yaml \
    cells/tm_tray/workflows/inspect_and_sort.yaml --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
    --out tools/kaggle/jobs/strict_full.json
# 3. on a GPU (Kaggle, env curobo-warm)
python tools/kaggle/curobo_free_plan.py --job tools/kaggle/jobs/strict_full.json --out results/strict_full_paths.json
# 4. time, gate, write (--paths cpu: plan the free moves here instead)
PYTHONPATH=src python tools/plan_workflow.py assemble cells/tm_tray/layouts/strict.yaml \
    cells/tm_tray/workflows/inspect_and_sort.yaml --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
    --paths results/strict_full_paths.json --out cells/tm_tray/plans/strict_inspect_and_sort.json
```

## Joint limits: 5° operating margin

The real robot has the URDF's limits (J1, J2, J4–J6 ±360°, J3 ±158°). The
layout's `robot.limit_margin_deg: 5.0` keeps everything planned inside them by
5°: IK, the walks, the CPU planner, the gate and the cuRobo URDF variant
(`curobo_free_plan.variant_urdf` moves the revolute limits in by the margin).

## One arm configuration per run (teach, P3 change)

P3 chose each sequence's IK branch on its own (largest worst-case slack). That
put, e.g., the incoming rack on elbow −1 and the inspection table on elbow +1:
carrying the tray between them flipped the elbow and the wrist (J1 −228°, J3
+208°, J4 +170° in one free move). Now:

* **branch**: `process.continuity.prefer.branch` (default `auto`) — the
  configuration (signs of J3, J5) every sequence can use, with the largest
  worst-case slack over all of them (`teach.choose_branch`). strict, workflow
  and whole cell: **J3 −1, J5 +1**;
* **whole turns**: J1/J4/J6 have ±355° of travel, so a pose has up to three
  joint values per joint. In a slot table each sequence is turned nearest
  `q_ref` (default `process.motion.home`); in a workflow the turns are chosen
  for the run: per joint, the sum over transitions of |q_end − q_start| is
  minimised exactly (`teach.align_turns`: the recurring sequences — the
  inspection table's pick and place — are enumerated, Viterbi over the run for
  the rest). Before: transitions with J6 turning 353° for nothing.

## Segments

`plan.segments_for_workflow` turns the workflow (scheduler, occupancy) and its
teach table into segments:

| kind | what | scene |
|---|---|---|
| `free` | home → first point, between runs of a sequence, carry → place, → home | every tray where it is now; the held tray attached to the flange |
| `line` | straight tool line (in the shelf) | the phase's scene and contact rules (from the skill) |
| `path` | the tilt (pivot + retreat) | its own 10 mm clearance |
| `hold` | jaw open/close, AprilGrid reads, wait | — |

One cycle of `inspect_and_sort`: 42 segments (16 line, 4 path, 7 free, 15 hold);
the full run (12 cycles): 493 (192 line, 48 path, 73 free, 180 hold).

## Timing

`process.motion` (layout). User 2026-11-28: a free move carrying the tray is as fast
as an empty one, and the lines in the shelf run at half the free moves' joint speed.

| caps | limits | used for |
|---|---|---|
| `in_jig` | `joint_scale: 0.5`: every joint ≤ 0.5 rad/s, 1.0 rad/s² | lines inside the shelf, the tilt |
| `empty` | tool v 0.25 m/s, a 1.5 m/s², ω 0.5 rad/s, α 3.0 rad/s² | sight_far → sight_near (locate) |
| joints | 1.0 rad/s, 2.0 rad/s² | free moves, with or without the tray |
| `carry` (optional, not in the tm_tray layouts since 2026-11-28) | the carried object's own Cartesian caps | every segment that carries it |

Until 2026-11-28 the layouts had D30's IN_JIG (tool 0.05 m/s, 0.25 m/s², 0.25 rad/s,
1.0 rad/s²) and D14's carry caps (0.25 m/s, 1.5 m/s², 0.5 rad/s, 3.0 rad/s², 4.0 m/s²
downward): half of a run was the slow lines in the shelf.

Lines and the tilt with Cartesian caps: a trapezoid along the tool path (position
and rotation caps), then a continuous IK walk at every 10 ms sample. With
`joint_scale`: the path is walked geometrically (1 mm / 0.5°), a trapezoid on its
joint arc length (the largest joint's travel) with the scaled joint caps gives the
path parameter at every sample, and the IK is walked again there, so every sample
lies on the tool path; slowed until finite differences keep the caps. Free moves: a
trapezoid along the joint path, scaled uniformly until every joint is within its
caps; the path rests only at sharp corners (direction change > 25°); with `carry`
caps a carried object additionally keeps them. Jaws: 0.05 m/s plus 0.5 s settle. A
waypoint's `meta.dwell` (s) holds the arm still after it (the locate reads:
`process.read_dwell`).

## Output: `rsf.trajectory/1`

```json
{"schema": "rsf.trajectory/1",
 "joint_names": ["joint_1", ..., "joint_6", "left_gripper_joint", "right_gripper_joint"],
 "t": [...], "q": [[...]], "qd": [[...]],
 "segments": [{"name", "kind", "t0", "t1", "caps", "event"}],
 "meta": {"layout", "workflow", "paths", "problems", "ok"}}
```

10 ms samples, rest to rest per segment, the jaws included. The real-machine
program format is the user's (deferred); this is its neutral input, along
with the teach table.

## Results

strict, `inspect_and_sort`, the whole run (12 cycles: 24 transfers + the
table in between), free moves planned on the CPU (`--paths cpu`):

| | |
|---|---|
| segments | 493 (192 line, 48 path, 73 free, 180 hold) |
| free moves planned | 73/73 (RRT-Connect + shortcut + smoothing, 1 mm extra margin) |
| **CPU gate** | **0 problems** (every sample, 0.01 rad, exact meshes, 5 mm / skill rules) |
| trajectory | **1095.9 s**, 80 964 samples (10 ms) |
| time in lines in the shelf (`in_jig`, 0.05 m/s) | 515.5 s (47 %) |
| free moves carrying the tray (`carry`) | 322.6 s |
| free moves without the tray | 144.8 s |
| the tilt (`path`) | 62.3 s |
| jaws, reads (`hold`) | 39.7 s |
| CPU time | 33 min (planning ~19 min, timing + gate 14 min) |

Half of the run is the slow lines inside the shelf: D30's IN_JIG caps. They
and the carry caps are layout parameters (`process.motion.caps`), to be raised
after trials on the real machine.

Before the 1 mm planning pad, one carry passed a rack leg at 4.8 mm (the gate
samples the timed path at other points than the planner's edge checks): the
gate caught it.

The summary (segments, timing, problems) is kept in
`cells/tm_tray/plans/strict_inspect_and_sort_cpu.summary.json`; the samples
(14.6 MB) are regenerated with the command above.

## cuRobo on Kaggle

`tools/kaggle/jobs/strict_full.json`: 73 free moves, 37 distinct scenes stored
once, two robot variants (jaws open; jaws on the tray with the tray as a link).
Environment `curobo-warm` (T4, torch 2.9.1, Warp 1.19), kthfan/curobo dev/main.

| run | result |
|---|---|
| 1 (2026-11-16) | 0/73: the script read the trajectory as 6 columns (it carries the 2 locked jaws too); carry moves refused, "start or end state in collision" (tray vs the jaws holding it) |
| 2 (2026-11-18) | **51/73**: without the tray 47/49, with the tray 4/24. Failures: the trajectory optimiser's result is not collision-free. The 24 carry moves planned again *without* the tray: 21/24 — the tray's sphere model (fitted to the box, it bulges past it; the carry ends are ~10 mm from the jig) is the main cause |

All 51 GPU paths pass the CPU gate. The full run with them and the CPU
fallback for the other 22 (`--paths cells/tm_tray/plans/gpu/strict_full_paths.json --fallback cpu`):
**0 problems, 1039.1 s** (CPU only: 1095.9 s). The 47 empty moves from cuRobo
take 77.5 s against 135.0 s from the CPU planner (smooth, no corners); the 4
carry moves are equal (the tray's caps decide). Summary:
`cells/tm_tray/plans/strict_inspect_and_sort_gpu.summary.json`.

### With cuRobo's exact mesh checks deciding (2026-11-18)

`make_planner`: `mesh_collision_check` (robot meshes vs the scene, 5 mm),
`mesh_self_collision_check` (link meshes vs each other; pairs from
`RobotBuilder.compute_mesh_collision_matrix` — it added none — plus the tray
against the links holding it) and `mesh_only_success` (the meshes, not the
spheres, decide success; the spheres still steer the optimisers; the graph
planner's start/end check stays sphere-based).

| run | planned | without / with tray | pass the CPU gate | full run |
|---|---|---|---|---|
| spheres decide | 51/73 | 47/49, 4/24 | 51/51 | 51 GPU + 22 CPU: 0 problems, 1039.1 s |
| **meshes decide + mesh self-collision** | **68/73** | 47/49, 21/24 | 64/68 | **64 GPU + 9 CPU: 0 problems, 983.2 s** |
| same, scene margin +1 mm (`--pad`) | 66/73 | 47/49, 19/24 | 65/66 | — |

The gate rejects: the tray 4.2–4.7 mm from a rack leg (between cuRobo's
25 ms samples), and with the pad one empty move with `gripper_body` 3.4 mm
from a jig part — more than sampling explains; possibly the per-link triangle
budget (20 000) simplifying the gripper mesh (unverified). Paths the gate
rejects are replanned on the CPU before timing (`--fallback cpu`), so the
trajectory never contains them. Still failing in cuRobo: two moves to a far
AprilGrid read and three carries onto the table (one of them fails without
the tray too: tight ends, not the tray model).

### Known failures and candidate fixes (deferred, 2026-11-19)

All of them are handled today by the CPU fallback (the trajectory passes the
gate); they are skipped for now and kept here to resume later.

| # | case | segments | cause | verified? | candidate fixes |
|---|---|---|---|---|---|
| F1 | cuRobo finds no path to the far AprilGrid read (empty) | `free007:pick:incoming/L2/C1/sight_far`, `free061:pick:incoming/L0/C2/sight_far` | trajectory optimiser's result not collision-free by the meshes | no — not yet examined | check the end pose's real clearance and how close J3 is to its limit (sight_far is the tightest pose, P3); more attempts / graph-planner seeds; a via pose; move the far read (`standoff_far`) |
| F2 | cuRobo finds no path for a carry onto the table, also without the tray | `free033:place:table/L0/C0/carry` | the ends are tight for the arm itself | partly — fails without the tray too | as F1; check the clearance at both ends; a via pose |
| F3 | cuRobo finds no path for a carry onto the table with the tray only | `free039`, `free045` (`place:table/L0/C0/carry`) | the tray's collision spheres bulge past the box and steer the optimiser away; carry ends ~10 mm from the jig | partly — succeed without the tray | tray spheres kept inside the box (own sphere fit) or attached as a cuboid via cuRobo's attach API; the carry ends' tilt clearance |
| F4 | the gate rejects cuRobo carry paths (tray 4.2–4.7 mm from a rack leg, 5.0 required) | `free027`, `free036`, `free069`, `free072` | cuRobo checks the meshes at its trajectory samples only (25 ms); the path dips between them | yes — the +1 mm pad removes all four | `--pad 0.001` (costs 2 planned carries: 66 vs 68), or a finer validation step / swept check in cuRobo |
| F5 | with the pad, the gate rejects one empty move (`gripper_body` 3.4 mm from a jig part, 6 mm asked) | `free019:pick:incoming/L2/C3/sight_far` | more than sampling explains; possibly the per-link triangle budget (`max_triangles_per_link: 20000`) simplifying the gripper mesh | no | raise the budget / compare the simplified mesh with the original (distance error); cross-check that pose with `get_mesh_collision_distance` |
| F6 | free carries tilt the tray off the plate (found in P5) | 8 of 24 carries > 30° from the pick tilt (cuRobo and CPU alike); `free051` 126°: the tray fell in simulation | neither planner constrained the tool's orientation while carrying | yes (P5 simulation, then the gate) | **done on the CPU side**: `carry_max_tilt_deg` (30°) in the CPU planner and the gate; GPU paths breaking it are replanned on the CPU. **cuRobo still to do**: an orientation constraint on the carry moves (`ToolPoseBound` on the tool's rotation, or the planner's pose-cost bound), so its paths pass the gate |
| F7 | carry timing ignored the tray's rotation (found in P5) | 2 carries above ω 0.5 rad/s | caps on the carried point's linear motion only | yes | done: angular speed and acceleration scale the carry timing |
| — | "Start or End state in collision" warnings (78 in one run) | many | the graph planner's start/end check uses the spheres only | yes (cuRobo docs) | not a failure: the optimiser still succeeds; would matter only if the graph planner were needed |

## Not yet

* Blending through via points / between segments (every segment is rest to
  rest; cuRobo's own time-optimal output could be kept for free moves).
* Real-time replanning (P7), simulation (P5), locating (P6).
* The other layouts' workflow runs (`as_built`, `recommended`): same commands.

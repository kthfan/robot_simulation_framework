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

## Timing (D14 / D30 caps)

`process.motion` (layout):

| caps | v (m/s) | a (m/s²) | ω (rad/s) | α (rad/s²) | used for |
|---|---|---|---|---|---|
| `in_jig` | 0.05 | 0.25 | 0.25 | 1.0 | lines inside the shelf, the tilt |
| `carry` | 0.25 | 1.5 | 0.5 | 3.0 | the carried tray's centre (free moves with the tray), a_down 4.0 |
| `empty` | 0.25 | 1.5 | 0.5 | 3.0 | sight_far → sight_near |
| joints | 1.0 rad/s | 2.0 rad/s² | | | free moves |

Lines and the tilt: a trapezoid along the tool path (position and rotation
caps), then a continuous IK walk at every 10 ms sample. Free moves: a trapezoid
along the joint path, scaled uniformly until every joint is within its caps;
the path rests only at sharp corners (direction change > 25°); a carried tray
additionally keeps its own Cartesian caps. Jaws: 0.05 m/s plus 0.5 s settle.

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

## Not yet

* Blending through via points / between segments (every segment is rest to
  rest; cuRobo's own time-optimal output could be kept for free moves).
* Real-time replanning (P7), simulation (P5), locating (P6).
* The other layouts' workflow runs (`as_built`, `recommended`): same commands.

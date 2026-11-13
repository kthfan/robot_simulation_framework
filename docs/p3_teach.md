# P3 — automatic teach points

Date: 2026-11-13. Engine: `src/rsf/planning/teach.py`. TM tray skills:
`cells/tm_tray/plugin/tm_tray/skills.py`. Output: `cells/tm_tray/teach/`.

## What it does

For every motion a task needs, one joint configuration per waypoint that is
reachable, collision-free, and continuous with its neighbours, plus the
numbers to judge it by.

```bash
# every slot, pick and place
PYTHONPATH=src python -m rsf teach cells/tm_tray/layouts/strict.yaml --skill tm_tray.pick tm_tray.place --out cells/tm_tray/teach/strict
# only what a workflow run needs, each motion once
PYTHONPATH=src python -m rsf teach cells/tm_tray/layouts/strict.yaml --workflow cells/tm_tray/workflows/inspect_and_sort.yaml --out cells/tm_tray/teach/strict_inspect_and_sort
```

Writes `<out>.json` (everything, plus the layout, process parameters and, for
a workflow, the task list and which sequences each task runs) and `<out>.csv`
(one row per point, degrees and millimetres). The real-machine program format
is the user's to design; these files are its neutral input.

## How (generic)

A skill turns a task into **sequences of waypoints**. A waypoint is a tool pose,
the scene at that moment (obstacles, attached objects), the contact rules,
extra joints (jaws), and how the tool *arrives*:

| arrive | meaning | checked |
|---|---|---|
| `free` | any collision-free joint motion (P4 plans it); the branch may change | end only |
| `line` | straight tool line (5 mm / 1° samples) | every sample, one continuous IK branch |
| `path` | given tool poses (the tilt) with its own clearance | every sample and the end |

The engine solves every waypoint for all exact in-limit IK branches, walks each
passing start branch through the line/path arrivals (Newton refinement from the
previous sample, falling back to full IK; joint step ≤ 0.2 rad, no elbow/wrist
sign change), collision-checks each sample with that waypoint's rules, and per
run of constrained arrivals keeps the chain with the **largest worst-case slack**
(clearance beyond the required margin), then the least joint travel. The
longest run anchors the sequence; the other runs take the chain nearest to it
in joint space.

Each **teach point** records: joints (rad), tool pose (xyz, quaternion),
commanded jaws, how it is reached, the clearance and the pair it is measured
between, the worst slack along the arrival, manipulability (Yoshikawa), the
smallest joint-limit margin, the arm configuration (signs of joints 3 and 5),
and the action done there.

New cells add skills (`register("skill", name)`); nothing in the engine is TM
specific. `process.continuity` sets which joints define the configuration.

## TM tray skills

| Sequence | Waypoints (arrival) | Jaws |
|---|---|---|
| `pick:<slot>` | sight_far (free), sight_near (line), pre_insert (free), insert (line), push (line), lift (line), withdraw (line), carry (tilt path) | open; closed to the tray width after lift (D5) |
| `place:<slot>` | carry (free), withdraw (tilt path, reversed), place_lift (line), place_down (line), retreat (line) | closed; opened above the columns before going down (D30) |
| `tm_tray.transfer` | pick at `from` + place at `to` | |

Scenes are the worst case for neighbours (every other slot holds a tray, as in
M3). While the plate lifts the tray off its columns or sets it down, the robot
may touch the target tray along that line; the carried tray is judged at both
ends.

## Results (2026-11-13)

| Layout | Sequences | Time (CPU, 4 cores) |
|---|---|---|
| strict | **74/74** (37 pick + 37 place, 481 points) | 599 s |
| as_built | **74/74** (481 points; min manipulability 0.0097, min limit margin 7.3°) | 676 s |
| recommended | **74/74** (481 points; min manipulability 0.0124, min limit margin 8.1°) | 594 s |
| strict, `inspect_and_sort` workflow (36 tasks, 12 cycles, ends with `request_refill`) | **26/26** distinct sequences (the table's pick/place taught once) | 203 s |

The tilt keeps 10 mm everywhere in all three layouts; its tightest point is
its start (`withdraw`), exactly 10.0 mm from the own jig's front columns.

strict, worst case per phase over all 37 slots:

| Phase | min clearance | min slack along arrival | min manipulability | min limit margin | binding pair |
|---|---|---|---|---|---|
| pick/sight_far | 15.35 mm | – | 0.0135 | 7.1° | link_2 – pedestal plate |
| pick/sight_near | 12.65 mm | 7.59 mm | 0.0282 | 19.3° | link_2 – pedestal plate |
| pick/pre_insert | 18.46 mm | – | 0.0230 | 21.6° | gripper body – own label |
| pick/insert | 2.00 mm (designed) | 0.78 mm | 0.0648 | 61.0° | push plate – tray |
| pick/push | 0 (designed contact) | 0 | 0.0640 | 62.1° | push plate – tray |
| pick/lift | 8.17 mm | 3.17 mm | 0.0577 | 61.3° | gripper body – jig above |
| pick/withdraw | 10.00 mm | 3.17 mm | 0.0225 | 20.4° | carried tray – own jig |
| pick/carry | 10.07 mm | 0 over 10 mm (start of the tilt) | 0.0274 | 28.2° | carried tray – jig above |
| place/place_lift | 8.17 mm | 3.17 mm | 0.0592 | 60.6° | gripper body – jig above |
| place/place_down | 0 (designed contact) | 0 | 0.0649 | 61.4° | push plate – tray |
| place/retreat | 18.46 mm | 0.78 mm | 0.0230 | 21.6° | gripper body – own label |

Arm configurations chosen (joint 3, joint 5 signs): (−,−) 28, (+,+) 28,
(+,−) 10, (−,+) 8 sequences — each sequence takes its most clear chain.

## Not yet

- **Free moves** between runs and between tasks (sight_near → pre_insert,
  pick → place) are only end-checked; P4 plans them. Sequences choose their
  arm configuration independently, so consecutive tasks may change it; P4 can
  prefer a configuration shared with the neighbours.
- **Timing** (speeds, accelerations, D14 carry caps) is P4.
- **sight_far** sits 7.1° from a joint limit at its tightest slot; harmless for a
  point, worth watching when P4 plans moves around it.

# Vocabulary, references and decisions

This page explains the short names the code and docs use, especially the ones
that point outside this repository. Some came from the predecessor project
**AI-Robotics-V2**: D-numbers, M-milestones and `findings/NN`. That project's
texts are **not in this repository**; its `cell_core` code is, frozen, in
`tests/reference/ai_robotics_v2`. Each entry below says what the reference
means *as it is used here*, so you do not need the original.

## Framework terms

| Term | Meaning | Code |
|---|---|---|
| cell | one robot installation: assets, layouts, workflows, plugin under `cells/<cell>/` | |
| World | a layout compiled: solids, features, objects, robot | `rsf.model.world.World` |
| solid | one collision shape at a world pose (box, mesh, cylinder, sphere), with tags | `Solid`, `Shape` |
| feature | a named frame (`kind`, `attrs`) that selectors match: a slot, a mount point | `Feature` |
| slot | a feature where an object can stand (`kind: slot`) | |
| object | something the robot moves; a *template* stands at each feature it may occupy | `ObjectInstance` |
| occupancy | `{feature key: object}` at some moment of a run | |
| generator / check / skill / scheduler / effect | registered extension points (`rsf.model.registry`) | docs/new_cell.md §5 |
| task | one workflow step with concrete arguments (`expand`) | `TaskInstance` |
| sequence | the waypoints one skill call needs at one place (`pick:<slot>`) | `rsf.planning.teach.Sequence` |
| waypoint | a tool pose + scene + contact rules + how it is reached | `Waypoint` |
| arrival `free` / `line` / `path` | free motion (planned later) / straight tool line / given poses | |
| run (of a sequence) | waypoints joined by `line`/`path` arrivals: one continuous IK branch | |
| teach point | the chosen joints at a waypoint, with clearance and margins | `TeachPoint` |
| branch / configuration | signs of the elbow and wrist joints (`continuity.sign_joints`) | `ContinuityRule` |
| whole turns | J1/J4/J6 travel ±355°, so a pose has up to three values per joint; the run picks the least travel | `teach.align_turns` |
| contact rule / policy | per (robot link, obstacle) pair: required clearance, or contact allowed | `rsf.collision` |
| slack | clearance beyond the required margin (negative: violation) | |
| segment | a piece of the trajectory: `line`, `path`, `free`, `hold` | `rsf.planning.plan.Segment` |
| caps | the limits a segment is timed to: Cartesian (v, a, ω, α, a_down) or `joint_scale` (a fraction of the free moves' joint caps) | `process.motion.caps` |
| gate | the CPU check of every trajectory sample with exact meshes; nothing passes without it | `plan.check_segment` |
| job | the free segments written for cuRobo on a GPU | `rsf.curobo_job/1` |
| transfer | a pick sequence followed by a place sequence, judged as one | `rsf.sim.judge` |
| seat | where the plan sets an object down; the judge's *placement* criterion is the error against it (`seat_mm`, `seat_deg`) | `rsf.sim.judge` |
| cycle / binding / end event | one repetition of a workflow's steps; the features a scheduler gives it (`$name`); why the run stopped (`sources_empty`, `max_cycles`) | `rsf.model.workflow` |
| `robot_seat` | tag of solids the arm's base may touch (its mount, the floor) | |
| setup (step) | a workflow step run once before the first cycle (task cycle -1) | `Workflow.setup` |
| locate tour | the setup step of the tray cell: every slot's far and near AprilGrid read, rack by rack, before anything is moved (user 2026-11-28) | `tm_tray.locate` |
| read | a hold in which a camera on the robot images a marker; in simulation rendered and detected, and the board pose compared with the truth | `rsf.sim.reads` |
| dwell | how long the arm holds still after a waypoint's action (`meta["dwell"]`; the reads: `process.read_dwell`) | |

## This repository's phases (P0–P7)

Phases of `docs/PLAN.md` §7; each has a results doc.

| Phase | What | Doc |
|---|---|---|
| P0 | package skeleton, SE(3), URDF loader | — |
| P1 | cell model: layout, World, registry, selectors, workflow, audit | cell_model.md |
| P2 | kinematics (EAIK) and collision (coal) backends, reach, the M3 check | p2_results.md |
| P3 | teach engine (automatic teach points) | p3_teach.md |
| P4 | paths: free moves (cuRobo / CPU), timing, the gate, the trajectory | p4_paths.md |
| P5 | simulation (MuJoCo, Isaac Sim) and the judge | p5_sim.md, p5_isaac.md |
| P6 | locating with AprilGrids (simulation) | p6_locating.md |
| P7 | live view, control, recording | live_view.md |

## AI-Robotics-V2 references

| Ref | Meaning as used here |
|---|---|
| **D1** | the inspect-and-sort process: incoming tray → inspection table → outgoing rack; a run ends by asking for a refill or a removal |
| **D2** | multi-view fusion for locating (an idea kept for later) |
| **D4** | the tray is retained by gravity while carried (it lies on the plate) |
| **D5** | close the jaws to the tray width ("width fit") after lifting |
| **D9** | the robot driver seam: simulation and the real robot behind one interface (ROS2 JSON trajectories) |
| **D11** | layout poses get uncertainties, corrected later by AprilGrid registration |
| **D14** | carry caps: speed/acceleration limits of the carried tray (removed from the layouts 2026-11-28, user: a carry is as fast as an empty move); the 15° tilt-back design value |
| **D16** | `as_built`: the cell as measured on site (a record: derive variants, do not edit it) |
| **D17** | tray / jig / plate geometry: rails on column tops, a 2.5 mm recess the plate slides under |
| **D18** | a camera-frame mix-up (hand-eye vs another camera) that moved every sighting pose; the lesson: name frames explicitly |
| **D19** | `recommended`: `as_built` with the mounting leg 100 mm longer |
| **D23** | `strict`: `as_built` with a 0.5285 m leg (the layout the tray cell plans in) |
| **D21** | the reach sweep is the authority on which slots are reachable |
| **D22** | one source for geometry: derive values, never transcribe them; a test of a cross-boundary claim must call across the boundary |
| **D24** | the close AprilGrid read: standoff 0.2768 m, plate clearance against the tray below |
| **D28** | the gripper (supplied 2026-10-06) and its pick sequence without the push plate |
| **D29** | the corrected gripper model (removed a stray mount offset from `tm5s.usd`) |
| **D30** | jaw values, 2 mm overtravel / place offset, the in-shelf speed caps (`in_jig`; replaced 2026-11-28 by half the free moves' joint speed, user), plate friction, and the five pass criteria of a transfer — the basis of `rsf.sim.judge` |
| **M3** | milestone: every slot's pick and place collision-free (ported as `tm_tray.collision_check`) |
| **M4** | milestone: AprilGrid locating; defines the board frame and the vertical reach limit |
| **M5** | milestone: scanning with the RealSense D455 (not used for locating) |
| **M6** | milestone: paths and transfers (D30 is its reference) |
| **findings/04** | reach sweep of `as_built`: 35 of 37 slots reachable |
| **findings/07–08** | EAIK missed reachable poses on the calibrated TM5S; fixed by `RobustIK` (`rsf.kinematics.eaik`) |

## The user's decisions

Decisions are recorded with their date where they apply: in the YAML or code
comment next to the value (`# user 2026-11-23: ...`), and collected in
`docs/PLAN.md` §9 (in Chinese). The main ones:

| Date | Decision | Where |
|---|---|---|
| 2026-11-12 | inputs: URDF, meshes, USD; real-robot program format deferred (the user designs it) | PLAN.md §9 |
| 2026-11-12 | offline teach and planning run without a GPU; real-time planning (GPU) later | PLAN.md §0 |
| 2026-11-12 | the air cylinder is installed (default model); jaws never fully close (`operating_range`) | asset_spec.yaml |
| 2026-11-12/13 | push-plate pick; the tilt is one motion keeping ≥ 10 mm clearance | tm_tray layouts |
| 2026-11-16 | joint operating margin 5° inside the URDF limits; free moves by cuRobo on a GPU (Kaggle), CPU gate decides | p4_paths.md |
| 2026-11-19 | tray height 130 mm; carry the tray with care (tilt bounded) | conftest.py, layouts |
| 2026-11-23 | 5° tilt-back; carries within 15° of level | as_built.yaml |
| 2026-11-25 | live view: server-side RTX, browser viewer, API first | live_view.md |

A new decision: put it next to the value it sets (with the date) and add a row
to PLAN.md §9.

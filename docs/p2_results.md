# P2 — kinematics and collision backends: results

Date: 2026-11-12. Layout `cells/tm_tray/layouts/strict.yaml` unless noted.

## What exists

| Piece | Module | Runs on |
|---|---|---|
| Analytic IK: EAIK fork + `RobustIK` (robot_arm_motion_planner) | `rsf.kinematics.eaik` | CPU |
| Kinematics protocol, world placement (`ArmKinematics`, `load_arm`) | `rsf.kinematics` | — |
| Reach sweep over any registered target provider | `rsf.planning.reach` | CPU |
| Exact-mesh collision: coal BVH triangle meshes + boxes, signed distance, broad phase, **contact rules** (per-pair margins, allowed contacts), attached objects, mesh self-collision | `rsf.collision`, `rsf.collision.coal_backend` | CPU |
| IK-branch continuity (generalised M3 chain) | `rsf.planning.chain` | CPU |
| World → cuRobo scene | `rsf.compile.curobo` | — |
| World → MuJoCo (`MjSpec`) | `rsf.compile.mjcf` | CPU |
| TM tray: grasp targets, M3 check port, CLI | `cells/tm_tray/plugin/tm_tray/{targets,sequence,collision_check}.py`, `cells/tm_tray/tools/check_cell.py` | CPU |

## Results

### Kinematics
- EAIK FK = URDF FK to 1e-16; IK recovers the source configuration in 100/100
  random poses, every solution exact to 1e-8 on the calibrated model.
- The assembled URDF's flange is already in tm5s.usd's convention (FK(0) =
  `USD_FLANGE_R0`), so cell_core's targets need no conversion.

### Reach (259 poses = 37 slots × 7 phases)
| Layout | Current sequence (D28) | Pre-D28 sequence (findings/04) |
|---|---|---|
| as_built (0.6765 m) | **37/37** | 35/37 — outgoing/L3/C0, C5 fail insert + lift (as published) |
| strict (0.735 m) | 37/37 | — |
| recommended (0.7765 m) | 37/37 | — |

The pre-D28 run reproduces the original project's published result exactly,
which pins the IK. **New:** with the D28 sequence (insert ~30 mm shallower,
lift 21.25 mm instead of 40) the as-built mount reaches every slot
kinematically; the `tm_tray.mount_height` audit threshold (0.733 m) predates
D28. Collisions on as_built were not checked yet.

### M3 collision (CPU, exact meshes)
Since 2026-11-12 the check covers the **pick and the place** of every slot, with
the target tray where each phase has it (seated, shifted, or carried).

| Gripper / sequence | strict | as_built | recommended |
|---|---|---|---|
| D29 gripper (`tm5s_egu50_no_air_cylinder.urdf`), D28 grasp | **PASS 37/37** (as D29 with cuRobo) | PASS 37/37 | — |
| Installed gripper (push plate), D28 insert depth | FAIL 0/37 (`insert`: plate 9.5 mm into the tray) | — | — |
| **Installed gripper, push-plate pick/place, 15° tilt (default)** | **PASS 37/37** | **PASS 37/37** | **PASS 37/37** |
| ... same, 0° tilt | PASS 37/37 | — | — |
| ... same, 30° tilt | PASS 37/37 | — | — |
| ... 15° tilt applied at the withdraw pose (no back-out) | FAIL 11/37 (`carry`) | — | — |

About 160 s per layout on the CPU (13 phases per slot).

#### The push-plate pick and place (user 2026-11-12)
The push plate is fixed. Tilt alone could not settle the tray, so the arm
pushes it with the plate; the plate shields the jaws' moving parts.

| Phase | Plate tip depth | Target tray |
|---|---|---|
| pick `insert` | 204.5 mm | seated, assumed placed 2 mm shallow; plate 2.0 mm short of it |
| pick `push` (jaws open) | 210.5 mm | pushed to 2 mm past its seat, touching the plate (contact allowed, no penetration) |
| pick `lift`, `withdraw`, `carry` | — | carried, registered against the plate |
| place `place_lift`, `place_down` | 206.5 mm | set down 2 mm shallower than its seat |
| place `retreat` | — | seated 2 mm shallow |

Parameters, `process.push_plate` in `cells/tm_tray/layouts/as_built.yaml`:
`margin` 2 mm (until the real plate is measured), `overtravel` 2 mm and
`place_offset` 2 mm (AI-Robotics-V2 D30). The push plate's position is read
from the robot URDF. Audit checks: the insert tip stays past the tray middle
(69.5 mm), the push ends before the rails leave the front columns (2.0 mm of a
derived 8.5 mm), the push stroke is positive (6.0 mm). Without the push plate
the sequence is cell_core's grasp exactly (tested).

#### Tilt: back out first
With tilt kept as insurance (15°, D14's design value), tilting at the withdraw
pose fails: the carried tray's back end overhangs the plate tip by ~50 mm and is
still 18 mm inside the shelf, so tilting lifts it into the label hanger of the
jig above (−0.5 mm; only the top layers and the table passed). This was
AI-Robotics-V2 M2's contingent check ("tilt does not foul the shelf above").
The carry now backs out 50 mm first, so the tray's back end is 10 mm in front
of the label plane (`TILT_CLEARANCE`), then tilts; 0°, 15° and 30° all pass.

**Pending measurement:** if the push plate shares tm5s.usd's stray gripper
offset, the real plate is 1.76 mm closer to the tray and 3.41 mm lower than
modelled. Measure the gap between the push plate's lower edge and the plate's
upper face: ~7.0 mm (model right) or ~3.6 mm (offset present).

### cuRobo cross-check (Kaggle T4, cuRobo dev/main `445ed42`)
`tools/kaggle/make_crosscheck_cases.py` writes 352 configurations (every IK
branch of every grasp phase of 7 slots, D29 gripper) with the CPU's signed
distance from the robot meshes to the phase's scene; `tools/kaggle/curobo_crosscheck.py`
builds a cuRobo model of the same URDF (`RobotBuilder`, no mesh decimation) and
asks `get_mesh_collision_distance` for the same quantity on a T4.

| | Cases | Agreement |
|---|---|---|
| Clear (0.001 < d < 0.1 m) | 252 | **252/252**, max difference **1.9e-7 m** (float32), same nearest link in all |
| Penetrating (CPU d <= 0) | 98 | 98/98 reported <= 0 by cuRobo |

So the CPU backend and cuRobo's exact mesh check see the same geometry. The
penetrating cases show a cuRobo limitation worth feeding back: its depth is
the deepest robot *vertex* inside an obstacle, so it reads **0 for 50 of the
98** (an edge or face crosses, no vertex inside, typical of the 8-vertex
gripper boxes) and 1-63 % (median 33 %) of the true depth for the other 48.
The pass/fail verdict is unaffected (cuRobo's intersection test is separate),
but the depth cannot rank how badly a configuration collides.

### as_built collision (D29 gripper)
PASS 37/37 (48 s). With the current sequence the as-built mount is clear of
collisions at every grasp pose as well as reachable. Path feasibility between
poses (M6/P4) is still unchecked, which is what D23 kept the raised mount for.

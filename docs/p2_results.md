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
| Gripper / rule | strict | as_built | recommended |
|---|---|---|---|
| D29 gripper (`tm5s_egu50_no_air_cylinder.urdf`) | **PASS 37/37** (as D29 with cuRobo) | PASS 37/37 | — |
| As installed, with push plate, D28 insert depth | FAIL 0/37 (`insert` everywhere) | — | — |
| **As installed, insert stops 2 mm before the push plate meets the tray (default since 2026-11-12)** | **PASS 37/37** | **PASS 37/37** | **PASS 37/37** |

About 50 s per layout on the CPU.

Why the D28 depth fails with the push plate: the plate spans flange z
0.117–0.124 m; at the D28 insert (plate tip 0.218 m behind the rack face) the
tray's front face is at flange z 0.1145, 9.5 mm behind the plate's tray-side
face.

**The rule (user 2026-11-12):** `insert_tip_depth` takes the shallowest of the
D28 limits and `PLATE_FZ_TIP + TRAY_FRONT_DEPTH - push_front_z - margin`, with
`push_front_z` read from the robot URDF's `air_cylinder_plate` collider
(`cells/tm_tray/plugin/tm_tray/tool.py`) and `margin` from the layout's
`process.push_plate.margin` (0.002). The tip stops at 0.2065 m (11.5 mm
shallower), 71.5 mm past the tray's middle; the carried tray's place on the
plate follows. A robot without the push plate keeps the D28 sequence exactly.
Tests: the exact meshes measure 2.000 mm between the push plate and the target
tray at insert; only `insert` and `lift` move, by 11.5 mm along the approach.

**Pending measurement:** if the push plate shares tm5s.usd's stray gripper
offset, the real plate is 1.76 mm closer to the tray and 3.41 mm lower than
modelled (real gap ~0.24 mm). Measure the gap between the push plate's lower
edge and the plate's upper face: ~7.0 mm (model right) or ~3.6 mm (offset
present). Then adjust the margin or the asset.

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

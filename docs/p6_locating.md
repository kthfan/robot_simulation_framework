# P6 — locating: AprilGrid reads (simulation first)

Date: 2026-11-19. Code: `src/rsf/sim/vision.py`, `cells/tm_tray/plugin/tm_tray/vision.py`,
`tools/sim_locate.py`. Tests: `tests/test_vision.py`.

Before a slot is picked or placed, the locating camera reads its AprilGrid
label twice (`sight_far`, then `sight_near`) to correct the slot's pose (the
teach points assume the layout; the real rack is where it is). P6 will close
that loop; this first step checks the reads themselves in simulation.

## What is simulated

| part | model |
|---|---|
| label | per slot, the printed AprilGrid on the label's robot-facing surface: 4 × 1 tags of 36h11, 20 mm, 7.5 mm apart, 5 mm white margin (112.5 × 30 mm, the jig's label board); tags `4·grid_id + 0..3` left to right, grid ids by `geometry.grid_ids` (incoming, outgoing, table; bottom shelf first) |
| camera | the EIH 5M locating camera on the flange: the hand-eye calibration (`hardware.CAM_*`), 2592 × 1944 px, intrinsics of the focus config nearest the read distance (`FOCAL_CONFIGS`: config 6 for both reads, 0.35 m and 0.277 m) |
| image | MuJoCo, offscreen (OSMesa on the CPU), grey; the labels as textured quads, the cell as the planning model |
| detector | the user's kthfan/AprilgridDetector (Rust, Python binding built with maturin), `detect_boards` with the camera intrinsics |
| truth | the board frame of AI-Robotics-V2 M4: origin at tag 0's top-left corner, x along the printed columns, y down the rows, z into the label |

`rsf.sim.vision` is generic (markers, OpenCV-style pinhole cameras, grey
renders); MuJoCo's principal-point convention (sensor centre minus the
offset) is pinned by rendering points against an OpenCV projection.

```bash
pip install maturin opencv-python-headless
(cd ../kthfan/aprilgriddetector && maturin build --release --out dist --manifest-path crates/aprilgrid-py/Cargo.toml \
    && pip install dist/aprilgrid_detector-*.whl)
MUJOCO_GL=osmesa PYTHONPATH=src python tools/sim_locate.py cells/tm_tray/layouts/strict.yaml \
    --teach cells/tm_tray/teach/strict.json --out reports/locate_strict.json [--images reports/locate_img]
```

## Results (strict, ideal images)

74/74 reads (37 slots × far + near) find their own board (the neighbouring
labels are in view at the far read; the board is chosen by grid id). Pose
error 0.04 mm / 0.07° everywhere — constant, so it is the sheet's
rasterisation, not the geometry: the chain camera calibration → label pose →
board frame → detector is consistent.

These are ideal images (sharp, uniformly lit, no noise). They verify the
geometry, not the reading under real conditions.

## Locating the racks and correcting the run (strict, 2026-11-23)

Code: `rsf.model.layout` (`displace`), `cells/tm_tray/plugin/tm_tray/geometry.py`
(`slot_at`), `tools/locate_slots.py`. Tests: `tests/test_locate.py`.

**The truth.** A layout entity can be displaced rigidly (`displace: {xyz, rpy,
about}` or `{matrix}`): its solids and features move, and so does everything
built on them (jigs, trays, labels). Features whose attributes encode their pose
are rebuilt by a `feature_move` hook for their kind; for a tm_tray slot,
`slot_at` recomputes approach, lateral offset, label-plane distance and height
from the moved jig pose, so every pick and place pose moves with the rack
(checked to 1e-12). The scenario
`cells/tm_tray/layouts/perturbed/strict_racks_moved.yaml` stands both racks of
strict where they would be after being wheeled in:

| rack | displacement (about its own centre) | slots moved by |
|---|---|---|
| incoming | 6 mm further, 4 mm sideways, 2 mm higher, turned 0.6° | 6.8–10.3 mm, 0.6° |
| outgoing | 5 mm closer, 5 mm sideways, turned −0.5° | 3.6–9.0 mm, 0.5° |

**Reading.** The robot knows only the nominal layout and its teach points. For
every slot of both racks, the cell as it stands is rendered at the slot's taught
far read pose; the board the detector finds, against where the nominal layout
puts it, gives the slot's correction (yaw and translation only: the racks
stand on the floor). The near read is taken at the taught near pose moved by
that correction (IK nearest the taught joints), and refines it.

| read | worst error of the slot's corrected pose (36 slots) |
|---|---|
| far, at the taught pose | 0.13 mm / 0.09° |
| near, at the corrected pose | 0.05 mm / 0.02° |

**Correcting.** Each rack is rigid: its floor move (yaw, x, y, z) is fitted to
the near reads of all its slots, and the corrected layout is the nominal one with
each rack displaced as estimated (written by `locate_slots.py`).

| rack | estimated | truth (same pivot) | worst jig error, all its slots |
|---|---|---|---|
| incoming | 0.6002°, (5.994, −4.200, 2.000) mm | 0.6000°, (6.001, −4.199, 2.000) mm | 0.008 mm / 0.0002° |
| outgoing | −0.4995°, (5.010, 4.834, 0.000) mm | −0.5000°, (4.999, 4.834, 0.000) mm | 0.015 mm / 0.0005° |

**The run.** The whole strict workflow (24 transfers), simulated in MuJoCo in the
displaced cell:

| plan | taught and planned in | result |
|---|---|---|
| the nominal plan | the nominal layout | **fails at the first pick**: incoming/L2/C0 is 10.3 mm / 0.6° off; inserting, the jaws hit the jig's columns (3.9 mm), the gripper body runs 15 mm into the jig, and the simulation diverges (0/24) |
| the corrected plan | the located layout (workflow re-taught 26/26, CPU plan 1087.4 s, gate 0 problems) | **24/24 pass**: worst placement 0.39 mm / 0.02°, slip 0.40 mm, tracking 0.45 mm, carry tilt 15.0° — as in the undisplaced cell (0.43 mm, 0.44 mm, 0.44 mm) |

```bash
MUJOCO_GL=osmesa PYTHONPATH=src python tools/locate_slots.py \
    --truth cells/tm_tray/layouts/perturbed/strict_racks_moved.yaml --nominal cells/tm_tray/layouts/strict.yaml \
    --teach cells/tm_tray/teach/strict.json --racks incoming outgoing \
    --out-layout reports/locate/strict_located.yaml --out reports/locate/strict_located.json
PYTHONPATH=src python -m rsf teach reports/locate/strict_located.yaml --workflow cells/tm_tray/workflows/inspect_and_sort.yaml \
    --out reports/locate/strict_located_inspect_and_sort
PYTHONPATH=src python tools/plan_workflow.py assemble reports/locate/strict_located.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --teach reports/locate/strict_located_inspect_and_sort.json --paths cpu --out reports/locate/strict_located_cpu.json
PYTHONPATH=src python tools/sim_workflow.py cells/tm_tray/layouts/perturbed/strict_racks_moved.yaml \
    cells/tm_tray/workflows/inspect_and_sort.yaml --plan reports/locate/strict_located_cpu.json --out reports/locate/sim.json
```

What this does not cover yet:

* Ideal images (sharp, uniformly lit): the reads' accuracy under real
  conditions is the next item (blur, noise, exposure, shadows, occlusion).
* The racks are located once, before the run, as rigid bodies on a level floor.
  A slot moved on its own (a jig re-seated, a bent shelf) or a rack not level
  (roll, pitch: `slot_at` refuses a jig that is not level) is not modelled.
* Reading during the run (each slot just before its pick or place, then
  re-planning the moves that follow) is not done; neither is a perturbation of
  the table.
* as_built and recommended: not run (the user asked for strict first).

## Locating first, in the run (2026-11-28)

User 2026-11-28: the whole run includes locating, every jig is located before
anything is moved, and a pick or place never reads again.

* **Workflow.** `inspect_and_sort` has a `setup:` step, `tm_tray.locate` on every
  slot (`[$incoming.*, $outgoing.*, $inspection.*]`): per slot a sequence
  `locate:<key>` with the far read (`sight_far`, reached free) and the near read
  (`sight_near`, a line), the arm still `process.read_dwell` (0.5 s, assumed) at
  each. The tour goes rack by rack, top shelf first, the columns alternating
  (`skills.locate_order`). A pick starts at `pre_insert`.
* **Reads in the simulation** (`rsf.sim.reads`, `process.reads: tm_tray.reads`):
  at the middle of every read hold the simulated EIH camera renders, the
  detector finds the boards, and the board pose (the camera's pose from the
  simulated arm, as the real cell has it from its joints) is compared with the
  cell's truth. `tools/sim_workflow.py` does it for both backends: MuJoCo
  (OSMesa/EGL) and Isaac (RTX: the labels as textured, self-lit quads, the camera
  on the flange's body; a USD camera's principal point is the image centre, so it
  renders a slightly larger image centred on the optical axis and the calibrated
  size is cropped out of it at whole pixels, the image's own principal point,
  within half a pixel of the calibrated one, going to the detector).
* **Correcting.** `--located OUT.yaml` fits each fixture's floor displacement to
  its near reads against `--nominal` (the layout taught in) and writes the
  corrected layout; teaching and planning the transfers in it is the second,
  offline stage (as above). Replanning inside the same run is not done.

strict, one cycle (`cells/tm_tray/plans/strict_cycle1.json.gz`), MuJoCo, ideal
images: **74/74 reads found**, board error ≤ 0.041 mm / 0.073°; fitted against the
same layout every rack and the table come out within 0.03 mm / 0.011° of where
they are (incoming 12 near reads, outgoing 24, table 1). Both transfers pass.

## Next

* Realistic images: the detector's own degradations (`tools/agsim/render.py`:
  blur, noise, exposure, shadows, occlusion), lighting; detection rate and
  pose error versus them; then the run above with those reads.
* Locating during the run, per slot; levelling (roll, pitch) of the racks.
* The real images: compare with the original M4 dataset when available.

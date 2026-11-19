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

## Next

* Realistic images: the detector's own degradations (`tools/agsim/render.py`:
  blur, noise, exposure, shadows, occlusion), lighting; detection rate and
  pose error versus them.
* Locating proper: perturb the slots (rack placement, levelling) in the
  simulated cell, read, correct the slot poses, re-teach / re-plan, and judge
  the transfer in simulation (the P6 milestone: still passes with the slot
  perturbed). Both modes: before the run, and during it.
* The real images: compare with the original M4 dataset when available.

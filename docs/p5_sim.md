# P5 — simulation: the planned run in MuJoCo, judged

Date: 2026-11-19. Code: `src/rsf/sim/{mujoco_cell,judge}.py`,
`tools/sim_workflow.py`, `tools/render_sim.py`. Tests: `tests/test_sim.py`.

P4 checks geometry: every sample collision-free, inside the limits. P5 checks
that the motion works physically: the push-plate pick, carrying the tray on
the plate, setting it down. It replays the planned joint trajectory in MuJoCo,
lets the trays move under contact, friction and gravity, and judges every
transfer by D30's five criteria.

```bash
# one cycle (2 transfers); --transfers 0: the whole plan
PYTHONPATH=src python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml \
    cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_inspect_and_sort_gpu.json \
    --transfers 2 --out reports/sim_strict_1cycle.json
# sensitivity: any SimParams field
    ... --set friction.robot=0.1 --set friction.object=0.05
# pictures (offscreen, needs OSMesa: apt-get install libosmesa6)
MUJOCO_GL=osmesa PYTHONPATH=src python tools/render_sim.py <layout> <workflow> --plan <plan> --transfer 1 --out docs/images/p5_transfer1
```

## The simulated cell

| part | model |
|---|---|
| geometry | `rsf.compile.mjcf.world_spec`: the planned world; trays only where the workflow starts them (12 in `incoming`) |
| tray | free body, 1.2 kg, three boxes (two rails, body with the 2.5 mm recess) |
| robot | MuJoCo's URDF import; collision meshes as convex hulls per part (body, plate, camera, air-cylinder plate and body, two parts per jaw) |
| arm control | gravity compensated; per joint position + velocity servo with integral action, tracking the trajectory's q and qd |
| jaws | position servos, 1e5 N/m, force limit 100 N (the URDF's effort); EGU50 is screw driven, stiff |
| friction | robot 0.4, trays 0.3, static parts 0.5; MuJoCo combines a pair by the max, so plate–tray = 0.4 (D30: 0.5 static / 0.4 dynamic, combine max; MuJoCo has one coefficient, the dynamic one is used) |
| contacts | robot–robot off (self-collision is the planner's job); `solref` 0.002 s, `solimp` 0.95/0.99, elliptic cones, `impratio` 10, `noslip` 10 iterations; timestep 1 ms |

### Two simulator settings that mattered

* **Soft contacts creep.** With MuJoCo's default contact softness, a tray
  resting on the plate at 18.7° (μ 0.4 > tan 18.7° = 0.34: it must stick) crept
  1.5 mm in 6 s with the arm at rest, and ~2 mm during a carry — enough to fail
  placement. Stiffer contacts (`solref` 0.002, `solimp` 0.95/0.99) and the
  noslip solver: 0.18 mm in 6 s.
* **Jaw stiffness.** At 2000 N/m a jaw yields 2 mm under 4 N: the tray slid
  2 mm sideways during carries although the jaws leave only 0.5 mm per side.
  At 1e5 N/m: within the clearance.

Both are properties of the simulator, not of the plan; they are parameters
(`SimParams`) and recorded in every report.

## Judge (D30 criteria, `rsf.sim.judge`, backend independent)

| # | criterion | limit |
|---|---|---|
| 1 | placement: the tray ends where the plan sets it down (2 mm shallower than the seat, `place_offset`) | 2 mm, 1° |
| 2 | slip: the tray relative to the flange while carried (end of `lift:act` → start of `place_down`) | 5 mm |
| 3 | contacts: the robot touches only the tray it transfers (and its base its mount) | none |
| 4 | tracking: flange vs plan in the jigs (`in_jig` segments) | 1 mm |
| 5 | caps: the carried tray's planned motion within the carry caps (D14) | v, a, ω, α, a_down |

Transfers are found from the trajectory's segment names (`pick:<from>/…`,
`place:<to>/…`); the tray each one moves is tracked from the workflow's initial
occupancy.

## Results

### strict, first simulation (plan: 64 cuRobo + 9 CPU free moves, 983.2 s)

First cycle (2 transfers):

| transfer | placement | slip | tracking | contacts | caps |
|---|---|---|---|---|---|
| incoming/L2/C0 → table/L0/C0 | 0.25 mm / 0.01° | 0.45 mm | 0.42 mm | ok | ok |
| table/L0/C0 → outgoing/L0/C0 | 0.24 mm / 0.02° | 0.19 mm | 0.39 mm | ok | ok |

Plate friction sensitivity (first cycle, trays 0.05):

| plate μ | result | worst placement | worst slip |
|---|---|---|---|
| 0.4 (D30) | 2/2 pass | 0.25 mm | 0.45 mm |
| 0.25 | 2/2 pass | 0.47 mm | 0.44 mm |
| 0.1 (the asset's Teflon) | 2/2 pass | 0.45 mm | 0.45 mm |

The carry does not rely on friction: tilted back 15°, the tray rests against
the push plate (the original project's 8 mm creep at Teflon friction predates
the push plate and the tilt-back).

Whole run (24 transfers), same plan: **20/24**. The simulation found two gaps
in P4, invisible to a geometric check:

* **Carry orientation.** Transfer 17's free carry (`free051`, cuRobo) rolled
  the gripper until gravity pointed 126° off the plate normal: the tray left
  the plate and fell (the next transfer then had nothing to pick). 8 of 24
  free carries tilted the tray more than 30° from how it was picked up; some
  stood it almost upright on the push plate. Neither planner constrained the
  tool's orientation while carrying.
* **Angular caps.** Two carries turned the tray at 0.54 / 0.61 rad/s (cap
  0.5): the carry timing capped its linear motion only.

Fixed in P4 (`docs/p4_paths.md` F6, F7): `process.motion.carry_max_tilt_deg`
(30°) in the CPU planner and the gate, the carried object checked against the
arm's own links, and angular speed/acceleration in the carry timing.

### After the fixes: the whole run, every layout

| layout | plan | free moves | gate | simulated | worst placement | worst slip | worst tracking |
|---|---|---|---|---|---|---|---|
| strict | 1086.8 s | 58 cuRobo + 15 CPU | 0 problems | **24/24 pass** | 0.50 mm / 0.02° | 0.46 mm | 0.42 mm |
| as_built | 1101.6 s | 73 CPU | 0 problems | ASBUILT |
| recommended | RECPLAN | 73 CPU | 0 problems | RECSIM |

Pictures of the first transfer: `docs/images/p5_transfer1_close.png` (close-up
at the end of every phase), `p5_transfer1_wide.png` (overview),
`p5_transfer1.gif` (whole transfer, ×4).

## Not yet / limits

* Physical parameters not measured: tray–jig and tray–plate friction (D30's
  values used), contact stiffness, jaw stiffness, servo gains. The sensitivity
  runs above vary friction only.
* Convex hulls: MuJoCo collides each mesh part as its convex hull; the gripper
  parts are close to convex, the robot links are not (contacts with links far
  from the gripper are judged conservatively).
* The AprilGrid reads are not simulated (no camera images yet; P6).
* Isaac Sim adapter: the report format (`ExecReport`) and the judge are
  backend independent; an Isaac backend would produce the same report.

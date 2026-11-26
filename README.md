# Robot Simulation Framework (`rsf`)

`rsf` turns a description of a robot cell into a verified robot program. The
description is:

- **assets**: URDFs and meshes;
- **layout**: where things stand;
- **workflow**: what the cell does.

From these it computes, without a GPU:

1. **teach points**: every waypoint's joints, collision-checked;
2. **one timed joint trajectory** for the whole run, with every sample
   re-checked on exact meshes;
3. **a physics replay and a verdict**: the run simulated in MuJoCo or Isaac Sim
   and judged transfer by transfer (placement, slip, contacts, tracking, speed
   caps, tilt).

Cells plug in through a small Python plugin; nothing in `src/rsf` is
specific to one cell.

```
assets + layout + workflow + plugin ─► World ─► audit ─► teach ─► plan ─► simulate ─► judge
                                                (CPU)    (CPU)    (CPU; cuRobo GPU optional)
```

Two cells are in the repository:

- **`cells/example_pick_place`**: the smallest complete cell, a TM5S with a
  parallel gripper moving blocks. Start here.
- **`cells/tm_tray`**: the real target, a TM5S tray-handling cell with 37 slots.
  It has a push-plate pick, a tilt-back carry and AprilGrid locating.

## Quick start (CPU only)

Linux, Python 3.10+ (3.12 recommended), `build-essential` (EAIK compiles C++),
and `libosmesa6` for offscreen pictures:

```bash
python3 -m venv ~/venvs/rsf && source ~/venvs/rsf/bin/activate
```

```bash
pip install -e ".[mesh,collision,eaik,mujoco,usd]" pytest
```

```bash
bash cells/example_pick_place/run.sh
```

The last command takes about 20 s. It runs audit, expand, teach, plan,
simulate and judge, and ends with `SIM PASS: 2/2 transfers`.

```bash
python -m pytest -q
```

The test suite takes a few minutes. Tests whose dependency is missing are
skipped with the reason, and `-m slow` runs the long ones. With Isaac Sim (RTX
GPU) see [docs/local_setup.md](docs/local_setup.md). `tools/setup_local.sh`
installs everything (`SKIP_ISAAC=1` for CPU only).

## Documentation

| If you want to | Read |
|---|---|
| build a new cell or workflow | [docs/new_cell.md](docs/new_cell.md) |
| know the layout / workflow keys | [docs/cell_model.md](docs/cell_model.md) |
| read or write the output files | [docs/formats.md](docs/formats.md) |
| decode a term or a reference (D30, M3, P4, "slack", "gate") | [docs/decisions.md](docs/decisions.md) |
| run with Isaac Sim, on one or two computers | [docs/local_setup.md](docs/local_setup.md), [docs/p5_isaac.md](docs/p5_isaac.md) |
| watch a run live, record it, debug it | [docs/live_view.md](docs/live_view.md) |
| understand a stage in depth (design, results) | [p2_results](docs/p2_results.md) (kinematics, collision), [p3_teach](docs/p3_teach.md), [p4_paths](docs/p4_paths.md), [p5_sim](docs/p5_sim.md), [p6_locating](docs/p6_locating.md) |
| see the plan, the user's decisions and what is open | [docs/PLAN.md](docs/PLAN.md) (Chinese) |
| work on this repository as a coding agent | [AGENTS.md](AGENTS.md) |

## Repository map

```
src/rsf/
  core/          SE(3) helpers (numpy only)
  model/         layout -> World, URDF, workflow, registry, selectors, audit
  kinematics/    EAIK analytic IK (+ RobustIK refinement)
  collision/     contact rules; coal (exact meshes) backend
  planning/      teach engine, Cartesian walks, RRT-Connect, timing, the trajectory
  compile/       World -> MuJoCo (MJCF), USD (Isaac), cuRobo scene
  sim/           MuJoCo and Isaac cells, the judge, AprilGrid rendering
  live/          live view server, viewer, control, recording
  orchestrator/  schedulers
cells/<cell>/    assets, layouts, workflows, plugin, teach tables, plan summaries
tools/           command-line stages (plan_workflow, sim_workflow, locate_slots, ...), Isaac and Kaggle runners
tests/           pytest; tests/reference/ is the frozen predecessor code the tray cell is checked against
docs/            the documents above; docs/images/ pictures they show
```

## Status

The offline pipeline is complete and verified on the tray cell. Every slot is
taught, every run is planned (cuRobo or CPU), and the runs are simulated in
MuJoCo and Isaac Sim and pass the judge. Live view is complete.

Not done yet:

- export to the real robot's program format (to be designed);
- real-time replanning;
- realistic AprilGrid images;
- measured physical parameters.

See [docs/PLAN.md](docs/PLAN.md) §7 and §9.

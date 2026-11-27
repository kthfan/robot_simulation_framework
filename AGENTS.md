# Working on this repository (for coding agents)

Read [README.md](README.md) first. It says what the framework does and which
document covers what. This file covers how to work here: commands, where
things are, and the rules this codebase keeps.

## Commands

Run from the repository root. Python code imports `rsf` from `src/`, so every
command needs `PYTHONPATH=src`. Pytest sets it itself (`pyproject.toml`).

```bash
python -m pytest -q                       # default suite (a few minutes); "slow" tests deselected
python -m pytest -q -m slow               # the long ones (full-cell teach, whole-run simulation)
python -m pytest -q tests/test_example_cell.py   # the pipeline end to end on the example cell (~15 s)
bash cells/example_pick_place/run.sh      # the same through the command-line tools (~20 s)
```

| Area you changed | Tests to run |
|---|---|
| `rsf.model` (layout, world, workflow, registry, audit) | `test_tm_tray_layout.py`, `test_workflow_audit.py`, `test_usd_model.py`, `test_example_cell.py` |
| `rsf.kinematics`, `rsf.collision` | `test_kinematics_reach.py`, `test_collision.py`, `test_m3_collision.py` |
| `rsf.planning` | `test_teach.py`, `test_plan.py`, `test_rrt.py`, `test_example_cell.py` |
| `rsf.compile`, `rsf.sim` | `test_compile_mjcf.py`, `test_usd.py`, `test_sim.py`, `test_vision.py`, `test_example_cell.py` |
| `rsf.live` | `test_live.py` |
| the tray cell's plugin or assets | `test_tm_tray_*.py`, `test_m3_collision.py`, `test_teach.py`, `test_locate.py` |

Tests skip with a reason when an optional dependency is missing (`eaik`,
`coal`, `mujoco`, `pxr`, `aprilgrid_detector`, `isaacsim`). A skip is not a
pass: say which tests were skipped and why. Offscreen rendering uses OSMesa
(`MUJOCO_GL=osmesa`, set by `tests/conftest.py`; `apt install libosmesa6`).
Isaac tests run only in Isaac Sim's Python (docs/local_setup.md). GPU work
(cuRobo, Isaac on Kaggle) goes through `tools/kaggle/` and the
`.claude/skills/isaac-sim-kaggle` skill. Kaggle credentials come only from the
environment variables `KAGGLE_USERNAME` / `KAGGLE_API_TOKEN`; never write them
into a file.

## Where to start for a task

| Task | Start at |
|---|---|
| a new cell, skill or workflow | docs/new_cell.md and `cells/example_pick_place/` (copy it) |
| a layout / workflow key, a scheduler | docs/cell_model.md; `src/rsf/model/layout.py`, `workflow.py`, `rsf/orchestrator/scheduler.py` |
| why a teach point fails | `rsf.planning.teach` (`TeachEngine._follow` writes the messages); the skill's scene and contact rules |
| a free move fails or the gate rejects a sample | `rsf.planning.plan` (`plan_free_cpu`, `check_segment`); docs/p4_paths.md |
| a simulated transfer fails | `rsf.sim.judge` (criteria), `rsf.sim.mujoco_cell` / `isaac_cell`; docs/p5_sim.md |
| the tray cell's motions | `cells/tm_tray/plugin/tm_tray/sequence.py` (phases), `skills.py`, `hardware.py` (measured constants) |
| an output file's fields | docs/formats.md |
| a short name (D30, M3, P4, "slack") | docs/decisions.md |

Every module starts with a docstring that states its contract. Read it
before changing the module.

## Rules this codebase keeps

1. **One source for every number.** Derive values from the assets and the
   layout (read the URDF, compute from features); do not copy a number into a
   second place (AI-Robotics-V2 D22). Generated files (URDFs from
   `tools/make_assets.py` / `tools/assemble_tm_tray_assets.py`) are changed
   by changing the generator and regenerating, not by hand.
2. **A check must be able to fail, and a claim must be tested across the
   boundary it crosses.** A new check gets a test where it fails
   (negative / mutant case). A port is compared against the original by
   calling the original (`tests/reference/`).
3. **Verdicts with reasons.** Stages report a verdict plus the numbers and
   the names involved (which link, which obstacle, how many mm). Never weaken
   a check, a margin or a test to get a pass. Report the failure.
4. **Strict inputs.** Loaders, generators and `process.*` readers reject
   unknown keys, so a typo fails instead of silently doing nothing.
5. **Frozen and recorded files.**
   - `tests/reference/` is a verbatim copy of the predecessor's code: never
     edit it. User corrections to its constants go in
     `tests/conftest.py:CORRECTIONS`.
   - `cells/tm_tray/layouts/as_built.yaml` is the measured cell: derive
     variants with `extends:` instead of editing it.
6. **The user's decisions** are recorded with their date next to the value
   they set (`# user 2026-11-23: ...`) and in docs/PLAN.md §9. Do not change
   such a value without the user. If a task conflicts with one, ask.
7. **Units**: metres, radians, seconds; quaternions `wxyz`; 4×4 homogeneous
   poses. Names with `_mm` / `_deg` are the exceptions.
8. **Generality**: nothing under `src/rsf` may name a cell. Cell-specific
   behaviour goes in the cell's plugin or its `process:` parameters, with
   the tray cell's values as defaults where the framework already had them.
9. **Outputs**: teach tables and plan summaries are committed; trajectories
   (`cells/*/plans/*.json`) and `reports/` are not (regenerable, large).
10. **Docs move with the code.** When you change a stage, update its doc.
    When you change a file format, update docs/formats.md. When you add a
    term, add it to docs/decisions.md. When you add a process parameter,
    update docs/new_cell.md §7.

## Pitfalls

- Isaac Sim's Python brings its own `pxr` and `cv2`. Never `pip install
  usd-core` or `opencv-python` into an Isaac venv or `python.sh`: a second `pxr`
  breaks Isaac's extensions at start. In the standalone `python.sh`, `pxr` exists
  only once the app runs. `rsf.model.usd_model` starts the app itself when it reads
  a USD model there; code that needs `pxr` earlier calls
  `rsf.sim.isaac_cell.start_app()` first.
- MuJoCo and PhysX collide meshes as convex hulls; the planner (coal) uses
  exact triangles. Concave parts must be split into convex pieces, or the
  simulation disagrees with the plan (the audit's `rsf.convex_meshes` warns).
- A USD file that does not author `metersPerUnit` / `upAxis` is centimetres and
  Y up (USD's fallbacks). Tests that write USD set both.
- A skill's first waypoint must arrive `free` (the engine rejects anything
  else).
- The audit runs every *loaded* plugin's checks unless the layout lists
  `audit: [...]`. In one test session several cells' plugins are loaded.
- `registry.register` refuses a second registration under the same name.
  Plugins are imported once per process (`registry.load_plugin`).

---
name: isaac-sim-kaggle
description: Run NVIDIA Isaac Sim (PhysX physics and RTX camera rendering) for this repository on a free Kaggle GPU (2 x T4) from a machine without a GPU, using the cached Isaac environment (Isaac Sim 6.2, RTX shader cache, Vulkan) in tools/isaac. Use it whenever a task needs Isaac Sim — replaying a planned trajectory in Isaac (`sim_workflow.py --backend isaac`), cross-checking MuJoCo against PhysX, rendering the cell or camera images with RTX, testing rsf.sim.isaac_cell or rsf.compile.usd on real Isaac, or keeping the Isaac cache current — even if the user just says "try it in Isaac", "render it", or "run the Isaac tests".
---

# Isaac Sim on Kaggle

This session has no GPU. Isaac Sim runs as a batch job on a Kaggle GPU kernel that clones the
**pushed** commit, restores a prebuilt environment, runs one shell command at the repository
root, and brings back the files you name. Each run costs about 4 minutes of setup plus your
command. Batch as much as you can into one run, because a run takes Kaggle quota and minutes
of wall time.

Everything lives in `tools/isaac/`. Run `tools/isaac/kaggle_isaac.sh` from the repository
root. It builds on the `kaggle-gpu-test` skill's `kaggle_run.sh`, which it finds by itself;
read that skill for exit codes and quota.

## Before a run

1. **Check locally what does not need Isaac.** `rsf.compile.usd` and the Isaac cell's name
   maps run on CPU with `usd-core`:
   ```bash
   PYTHONPATH=src python -m pytest -q tests/test_usd.py
   ```
2. **Commit and push.** Kaggle sees only the pushed commit. Large inputs must be in git too.
   Plans are git-ignored: commit a trimmed `.json.gz`, which `sim_workflow.py` reads.
3. **Kaggle credentials** come from the environment (`KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`).
   Never print or store them.

## Commands

```bash
tools/isaac/kaggle_isaac.sh status                      # what the cache holds (driver, size, date)
tools/isaac/kaggle_isaac.sh run --cmd "CMD" --collect "reports/isaac/*" --timeout 60m [--slug NAME]
tools/isaac/kaggle_isaac.sh run --detach --cmd "CMD" --slug NAME   # then: fetch --slug NAME
tools/isaac/kaggle_isaac.sh rebuild [--wait]            # rebuild the cache (GPU, ~25 min)
```

`run` blocks until the job finishes, then prints a verdict, timings and the log tail. The full
log and the collected files are in `.kaggle-gpu-test/out/<slug>-<time>/`. Use a distinct
`--slug` for each concurrent job. Kaggle allows 2 GPU sessions at once and queues the rest.

Useful `CMD`s:

| goal | CMD |
|---|---|
| everything at once: probe, Isaac tests, strict first cycle in Isaac | `bash tools/isaac/kaggle_check.sh` |
| replay a plan and judge it | `PYTHONPATH=src python tools/sim_workflow.py <layout> <workflow> --plan <plan.json.gz> --backend isaac --transfers N --report reports/isaac/run.npz --out reports/isaac/run.json` |
| Isaac unit tests | `python -m pytest -q tests/test_isaac.py` (add `-m slow` for the cycle) |
| an RTX image of a stage | `python tools/isaac/render_probe.py --usd <stage.usda> --png reports/isaac/x.png` |
| Vulkan + render check | `bash tools/isaac/kaggle_vulkan.sh` |

Chain several with `;` in one `--cmd` rather than starting several runs.

## Writing Isaac code that runs there

- **Start the app before importing any Omniverse module.** Create
  `SimulationApp({"headless": True})` before importing `omni.*`. Isaac's own `pxr` also
  loads with the app. `rsf.sim.isaac_cell.IsaacCell` and `app()` do this for you.
- **`app.close()` ends the Python process.** Write results first: anything after it never
  runs.
- **Physics**: `IsaacCell(world, occupancy).run(traj)` gives the same `ExecReport` as
  MuJoCo, judged by `rsf.sim.judge`. Its parameters are `IsaacParams`, a `SimParams` with
  PhysX's solver fields. For your own scenes, use `rsf.sim.isaac_cell.start_physics(ctx, dt)`.
  Isaac's tensor view needs the stage id, and that function passes it.
- **Scenes**: `rsf.compile.usd.world_stage(world, path)` writes the cell as USD. Do not use
  Isaac's URDF importer: the compiler writes the robot itself, so the frames are the
  planner's. Prims carry `customData` `rsf:name` and `rsf:kind`, which map them back to the
  model.
- **Rendering**: Replicator (`rep.create.camera`, `render_product`, the `rgb` annotator). The
  stage has no lights, so add a dome light or a distant light. Use
  `tools/isaac/render_probe.py` as a template.
- **No EAIK, coal or MuJoCo** is installed in the Isaac environment, only numpy, pyyaml,
  trimesh, pillow and pytest. If an import chain pulls one in, make it lazy or add the
  package to `tools/isaac/kaggle_setup.sh` and rebuild.

## The cache, and keeping it current

The env kernel `kgt-env-isaac62` holds:

- **`isaac-venv.tar.zst`**: the 25 GB venv, 10.7 GB compressed. It includes Isaac Sim 6.2
  and its RTX shader cache, which is compiled for the T4 and this driver.
- **`nvidia-vk.tar.zst`**: Vulkan for that driver.
- **`env.json`**: records which driver it was built for.

`tools/isaac/kaggle_env_restore.sh` unpacks the cache in about 2.5 minutes. Without the
cache, installing Isaac takes about 10 minutes and the first RTX frame another 10 minutes.

**Kaggle has no Vulkan by default.** It exposes the GPU for compute only.
`tools/isaac/vulkan_setup.sh` adds the user-space graphics libraries of the running driver,
taken from NVIDIA's own installer, and registers:

- the Vulkan ICD;
- NVIDIA's EGL vendor for glvnd. This is the non-obvious part: without it,
  `vkCreateInstance` fails with `ERROR_INCOMPATIBLE_DRIVER`.

If Vulkan fails, physics still works, but there are no camera images.

**Driver changes.** If Kaggle's NVIDIA driver is not the cached one:

1. The restore reinstalls Vulkan for the new driver, so the run still works. The first RTX
   frame will recompile shaders for about 10 minutes.
2. It writes `ISAAC_ENV_STALE`.
3. `kaggle_isaac.sh` sees the marker after `run` or `fetch` and starts `rebuild` (detached).
   Set `KGT_ISAAC_AUTOREBUILD=0` to stop it.

Also rebuild when you change `tools/isaac/kaggle_setup.sh`, for example to pin a new Isaac
version or add a package.

## Things that cost time if you don't know them

- **Never `fetch` the env kernel** (`kgt-env-*`): its output is the 10+ GB archive. Use
  `kaggle_isaac.sh status`, which downloads only `env.json`.
- **Poll Kaggle sparingly.** Wait at least 2 minutes between status calls. Polling every
  30–60 s across several jobs gets HTTP 429.
- **"Maximum batch GPU session count of 2 reached"** means two GPU kernels are running or
  queued. Wait. The API cannot cancel a kernel, only the Kaggle web UI can.
- **The judge does not need EAIK.** It computes planned flange poses from the URDF.
  `sim_workflow.py --report` saves the raw run before judging, so a judging error does not
  waste the simulation.
- **Expected log noise.** Lines about audio devices, GLFW, "failed to open the default
  display" and `frictionType is deprecated` are harmless.
- **Expected differences from MuJoCo.** Placement and slip differ by up to about 0.6 mm
  because the contact models differ. Tracking and tilt agree to 0.01 mm and 0.01°. See
  `docs/p5_isaac.md` for the design, the measured numbers and the reasons.

---
name: kaggle-isaac-sim
description: Run NVIDIA Isaac Sim 6 (PhysX physics, articulated robots, contacts, and RTX camera rendering with Replicator) for any public GitHub repository on a free Kaggle GPU (2 x T4) from a machine without a GPU, such as Claude Code on the web or a CPU sandbox. A cached environment (Isaac Sim venv, warm RTX shader cache, Vulkan for Kaggle's driver) gets a run going in about 4 minutes instead of 20, and is rebuilt automatically when Kaggle's driver changes. Use it whenever a task needs Isaac Sim or Omniverse physics or rendering — simulating or replaying a robot or scene in Isaac, cross-checking another simulator against PhysX, rendering USD scenes or synthetic camera images, testing code that imports isaacsim or omni — even if the user only says "try it in Isaac", "render the USD" or "run the Isaac tests", or just notes there is no GPU here.
---

# Isaac Sim on Kaggle

The session has no GPU, so Isaac Sim runs as a batch job on a Kaggle GPU kernel:

1. It clones the **pushed** commit of the current repository.
2. It restores the cached Isaac environment.
3. It runs one shell command at the repository root.
4. It brings back the files you name.

Each run costs about 4 minutes of setup plus your command, and uses Kaggle GPU quota
(about 30 h a week). Batch work into as few runs as you can.

Everything goes through `scripts/isaac_kaggle.sh` in this skill's folder. Call it by its
absolute path, from the root of the repository. It bundles its own Kaggle runner
(`scripts/kgt/`) and ships the Kaggle-side scripts (`scripts/remote/`) with every job, so the
repository needs no changes.

## Requirements (check once)

- **Kaggle CLI**: `pip install kaggle`.
- **Credentials**: `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`, or
  `~/.kaggle/kaggle.json`). The account must be phone-verified to use GPUs and the internet.
  Never print or store credentials.
- **Network**: the sandbox must be allowed to reach kaggle.com.
- **Repository**: a **public** GitHub repository, with the commit you want to test pushed.
  Large inputs must be committed too. Kaggle sees only what is pushed.
- **Check**: `isaac_kaggle.sh check` verifies all of the above without using quota.

## Commands

```bash
S=<this skill's folder>/scripts
$S/isaac_kaggle.sh status        # does the cache exist? which driver, Isaac version, date
$S/isaac_kaggle.sh rebuild       # build the cache: needed once, ~30 min of GPU plus queueing
$S/isaac_kaggle.sh run --cmd "CMD" --collect "reports/isaac/*" --timeout 60m [--setup "pip install -e ."] [--slug NAME]
$S/isaac_kaggle.sh run --detach --cmd "CMD" --slug NAME      # then: $S/isaac_kaggle.sh fetch --slug NAME
```

**First use.** Run `status`. If it shows no `env.json`, run `rebuild` and wait for it.
`status` returns `env.json` when the cache is ready. A `run` without a cache fails with
"isaac-venv.tar.zst not found".

**`run`.** It blocks until the job ends, then prints a verdict, the timings and the log
tail. The full log (`test_output.log`) and the collected files are in
`.kaggle-gpu-test/out/<slug>-<time>/`. Exit codes:

| code | meaning |
|---|---|
| 0 | passed |
| 1 | the command failed |
| 2 | usage problem |
| 3 | Kaggle or infrastructure problem |
| 4 | gave up waiting |
| 5 | the command hit `--timeout` |

On Kaggle, CMD runs with:

- the cached Python 3.12 venv (with `isaacsim`) first on `PATH`;
- `OMNI_KIT_ACCEPT_EULA=YES`;
- `ISAAC_KIT=/tmp/isaac-kit`, which holds this skill's `scripts/remote`, also on `PYTHONPATH`.

`--setup` runs after the restore. Use it for the project's own install. Keep it light: it
runs on every job.

Useful CMDs (chain several with `;` rather than starting several runs):

| goal | CMD |
|---|---|
| does physics work (falling box, tensor API) | `python $ISAAC_KIT/probe.py --out reports/isaac/probe.json` |
| inspect a stage: articulations, DOFs, links, bodies | `python $ISAAC_KIT/probe.py --usd scene.usda --out reports/isaac/probe.json` |
| an RTX image of a stage | `python $ISAAC_KIT/render.py --usd scene.usda [--look-at /World/robot] --png reports/isaac/x.png --out reports/isaac/x.json` |
| your own script or tests | `python tools/my_isaac_job.py` or `python -m pytest -q tests/test_isaac.py` |

`--collect` takes globs relative to the repository; up to 200 MB comes back. Kaggle's own
output directory, `/kaggle/working` (`$KGT_OUT`), also comes back whole.

## Writing Isaac code that runs there

Read `references/isaac_api.md` before writing physics, contact or rendering code. It has
working patterns for each. The rules that cost the most time when broken:

- **Create `SimulationApp` before importing any `omni.*` module or `pxr`.** Use
  `isaac_util.app()`.
- **`SimulationApp.close()` ends the Python process.** Write the results first, with
  `isaac_util.report()`. A script that saves after `close()` saves nothing.
- **Step physics with `isaac_util.start_physics(ctx, dt)`** and `physx.update_simulation`.
  The tensor view needs the stage id, which that helper passes.
- **Authored stages usually have no lights.** Add a dome light before rendering (`render.py`
  does).
- **The environment holds only Isaac Sim plus numpy, pyyaml, trimesh, pillow, pytest and
  zstandard.** If the code needs more, either:
  - install it in `--setup` (small packages), or
  - rebuild with `ISAAC_EXTRA_PIP="numpy pyyaml trimesh pillow pytest zstandard yourpkg"`.

  Never install `usd-core` there: its `pxr` conflicts with Isaac's.
- **Test locally first whatever doesn't need Isaac.** Write USD stages with `usd-core` on CPU,
  apply PhysX schemas by name (see the reference), and unit-test them. Use Kaggle only for
  what needs the GPU.

## The cache, and keeping it current

The cache is the output of the Kaggle kernel `kgt-env-<name>` (default `kgt-env-isaac62`;
choose another with `ISAAC_KAGGLE_ENV`). It holds:

- **`isaac-venv.tar.zst`**: the Isaac Sim venv (~25 GB, ~11 GB compressed), including the RTX
  shader cache compiled for the T4 and the build's driver;
- **`nvidia-vk.tar.zst`**: Vulkan for that driver;
- **`env.tar`**: the uv Python;
- **`env.json`**: what was built and for which driver.

The restore takes about 2.5 minutes. Without the cache, installing Isaac takes about 10
minutes and the first RTX frame another 10 minutes.

**Vulkan.** Kaggle exposes its GPU for compute only (`NVIDIA_DRIVER_CAPABILITIES=compute,utility`),
so there is no Vulkan and no RTX by default. `scripts/remote/vulkan_setup.sh` adds the
graphics libraries of the running driver, taken from NVIDIA's own `.run` installer, and
registers them:

- the Vulkan ICD;
- NVIDIA's EGL vendor for glvnd. This is the non-obvious part: without it,
  `vkCreateInstance` fails with `ERROR_INCOMPATIBLE_DRIVER`.

If Vulkan fails, physics still works, but there are no images.

**Driver changes.** If Kaggle's driver differs from the cached one:

1. The restore reinstalls Vulkan for the new driver, in about 15 s, so the run still works.
   Its first RTX frame recompiles shaders for about 10 minutes.
2. It marks the cache stale (`ISAAC_ENV_STALE`).
3. `isaac_kaggle.sh` sees the mark after `run` or `fetch` and starts a `rebuild` (detached).
   `ISAAC_AUTOREBUILD=0` turns this off.

Also rebuild after changing `ISAAC_VERSION` or `ISAAC_EXTRA_PIP`.

**A cancelled or failed rebuild leaves the old cache in use.** Runs mount the latest
*completed* version of the env kernel. `status`, however, reads the latest version, so after
a cancelled or failed rebuild it shows no `env.json` even though runs still work.

## Things that cost time if you don't know them

- **Never `fetch` the env kernel.** Its output is the 10+ GB archive. Use `status`, which
  downloads only `env.json`. The wrapper refuses `kgt-env-*` slugs.
- **Poll Kaggle sparingly.** Wait at least 2–3 minutes between status calls. Polling every
  30–60 s across several jobs gets HTTP 429.
- **"Maximum batch GPU session count of 2 reached"** means two GPU kernels are running or
  queued. Wait. The API cannot cancel a kernel, only the Kaggle web UI can
  (`https://www.kaggle.com/code/<user>/<slug>`).
- **Queues.** GPU jobs can wait in the queue from minutes to hours. For long jobs, use
  `--detach` and `fetch` later instead of blocking a session.
- **Use a distinct `--slug` for each concurrent job.** The same slug pushes a new version of
  the same kernel.
- **Save raw results before post-processing.** If analysis code fails after a 20-minute
  simulation, the raw arrays should already be on disk and collected.
- **Expected log noise.** Audio devices, GLFW, "failed to open the default display" and
  `frictionType is deprecated` are harmless.

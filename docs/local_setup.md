# Running locally with Isaac Sim

One Python 3.12 virtual environment holds Isaac Sim 6.2 and everything this
repository uses. The same code that runs on Kaggle runs there, with your own GPU.

Verified (2026-11-24):

* **The repository at Isaac Sim's pinned versions.** The whole dependency set
  (Isaac Sim 6.2.0.0 plus this repository's packages) resolves for Python 3.12
  on Linux with no conflict. Isaac Sim pins:
  * numpy 2.3.1, mujoco 3.12.0, torch 2.11.0;
  * OpenCV (`opencv-python-headless-noffmpeg`);
  * `pxr` through `usd-exchange`.

  At those versions, without Isaac Sim itself, the test suite passes (147
  passed), and the first strict cycle replays in MuJoCo with the same results
  as before.
* **Isaac Sim itself** was installed with pip on Linux (Kaggle, Python 3.12):
  * physics: the strict first cycle, 2/2;
  * RTX rendering.
* **`tools/setup_local.sh`** was tested end to end except the Isaac Sim step
  (`SKIP_ISAAC=1`).
* **Not tested:** Windows, and a desktop GPU.

## Requirements

| | |
|---|---|
| OS | Linux x86-64 with glibc ≥ 2.35 (Ubuntu 22.04 / 24.04; `ldd --version`). Windows is supported by Isaac Sim but this repository is untested there |
| GPU | NVIDIA RTX GPU (RTX rendering needs RT cores); driver per NVIDIA's [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). Kaggle ran driver 580.178.04 on a T4 |
| Python | 3.12 exactly (`python3.12`, `python3.12-venv`, `python3.12-dev`) |
| Disk | ≥ 50 GB free: the venv is ~25 GB, plus shader and extension caches |
| Build tools | `build-essential` (EAIK compiles C++17), `git`; Rust (`rustup`) for the AprilGrid detector |

```bash
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh     # Rust, for the detector
```

## Install

```bash
git clone https://github.com/kthfan/robot_simulation_framework.git
cd robot_simulation_framework
git checkout claude/robot-simulation-planning-arch-va45i3
bash tools/setup_local.sh ~/venvs/rsf-isaac        # TORCH_CUDA=cu130 for CUDA 13
source ~/venvs/rsf-isaac/bin/activate
```

The script installs, in this order:

1. **PyTorch 2.11.0** (CUDA build).
2. **Isaac Sim** (`isaacsim[all,extscache]==6.2.0.0` from pypi.nvidia.com).
3. **This repository**, editable (`pip install -e ".[mesh,collision,eaik,mujoco]"`):
   coal, trimesh, MuJoCo, kthfan/EAIK (compiled), motion_planner, urchin.
4. **The AprilGrid detector**: kthfan/aprilgriddetector, built with maturin
   into a cp312 wheel. It is cloned next to this repository if it isn't
   already there.

It ends by printing the versions it found.

The same steps by hand, if you prefer:

```bash
python3.12 -m venv ~/venvs/rsf-isaac && source ~/venvs/rsf-isaac/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
pip install -e ".[mesh,collision,eaik,mujoco]" pytest maturin
git clone https://github.com/kthfan/aprilgriddetector ../kthfan/aprilgriddetector
(cd ../kthfan/aprilgriddetector && maturin build --release -i python --out dist --manifest-path crates/aprilgrid-py/Cargo.toml)
pip install ../kthfan/aprilgriddetector/dist/aprilgrid_detector-*-cp312-*.whl
```

**Do not install into this venv:**

* `usd-core`, or the `usd` or `test` extras, which include it: Isaac Sim brings
  its own `pxr`, and two copies conflict.
* `opencv-python` or `opencv-python-headless`: Isaac Sim brings its own `cv2`.
* Other versions of `numpy`, `mujoco` or `torch`: Isaac Sim pins them, and
  NVIDIA asks for a dedicated environment for that reason.

## Environment variables (each shell)

```bash
source ~/venvs/rsf-isaac/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES          # Isaac Sim's EULA; without it the first import asks
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # MuJoCo's offscreen rendering on the GPU (the tools default to osmesa, the CPU)
export PYTHONPATH=src                    # optional: the editable install already finds rsf
```

## Check

```bash
python -m pytest -q                      # ~5 min; with Isaac Sim installed, tests/test_isaac.py runs too (GPU)
python -m pytest -q -m slow tests/test_isaac.py   # the strict first cycle in Isaac (PhysX)
python tools/isaac/probe.py --out reports/isaac/probe.json                  # Isaac starts, a box falls, tensor API
```

The first Isaac Sim start takes a few minutes (extension and shader caches).
The first RTX frame compiles shaders for about 10 minutes, once per driver.

## Running the workflow (strict; as_built / recommended the same way)

Teach tables are in git (`cells/tm_tray/teach/`). Full plans are not
(`cells/*/plans/*.json` are regenerated; only summaries and
`strict_cycle1.json.gz` are kept). So plan first.

```bash
L=cells/tm_tray/layouts/strict.yaml
WF=cells/tm_tray/workflows/inspect_and_sort.yaml

# 1. teach (optional: the tables are in git). Every slot (~13 min), then the workflow run (~4 min)
python -m rsf teach $L --skill tm_tray.pick tm_tray.place --out cells/tm_tray/teach/strict
python -m rsf teach $L --workflow $WF --out cells/tm_tray/teach/strict_inspect_and_sort

# 2. plan the run on the CPU: free moves (RRT-Connect), timing, the collision gate (~45 min)
python tools/plan_workflow.py assemble $L $WF --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
    --paths cpu --out cells/tm_tray/plans/strict_inspect_and_sort_cpu.json

# 3a. simulate and judge in MuJoCo (~8 min)
python tools/sim_workflow.py $L $WF --plan cells/tm_tray/plans/strict_inspect_and_sort_cpu.json \
    --out reports/sim_strict.json

# 3b. the same in Isaac Sim (PhysX); --transfers N stops after N transfers
python tools/sim_workflow.py $L $WF --plan cells/tm_tray/plans/strict_inspect_and_sort_cpu.json \
    --backend isaac --usd reports/isaac/strict.usda --report reports/isaac/run.npz --out reports/isaac/sim_strict.json

# 4. an RTX image of the cell (needs a stage, e.g. the one step 3b wrote)
python tools/isaac/render_probe.py --usd reports/isaac/strict.usda --png reports/isaac/strict.png \
    --out reports/isaac/render.json
```

P6 locating: rack displaced, located from AprilGrid reads, run corrected. See
`docs/p6_locating.md` for the whole sequence.

```bash
python tools/locate_slots.py --truth cells/tm_tray/layouts/perturbed/strict_racks_moved.yaml \
    --nominal $L --teach cells/tm_tray/teach/strict.json --racks incoming outgoing \
    --out-layout reports/locate/strict_located.yaml --out reports/locate/strict_located.json
```

### Looking at the cell in Isaac Sim's window

The replays run headless. `--set headless=false` opens a window, but the replay
loop steps PhysX directly and does not refresh it, so live viewing is not
supported yet. To look at the cell, open the stage step 3b wrote in the full
app:

```bash
isaacsim            # the full Isaac Sim app (isaacsim.exp.full); then File > Open > reports/isaac/strict.usda
```

The stage has no lights: add a dome light, or use the render probe.

## Optional: cuRobo on the local GPU

The free moves can be planned by cuRobo instead of the CPU, which is much
faster. cuRobo (kthfan/curobo, dev/main) needs its own PyTorch and Warp, so
keep it in a **separate venv** rather than the Isaac one. Install it per that
repository's README, with this repository installed editable.

```bash
# in the Isaac / CPU venv: write the job
python tools/plan_workflow.py job $L $WF --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
    --out tools/kaggle/jobs/strict_full.json
# in the cuRobo venv: plan it
python tools/kaggle/curobo_free_plan.py --job tools/kaggle/jobs/strict_full.json --out results/strict_full_paths.json
# back in the Isaac / CPU venv: time, gate (replans on the CPU what cuRobo got wrong), write the run
python tools/plan_workflow.py assemble $L $WF --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
    --paths results/strict_full_paths.json --fallback cpu --out cells/tm_tray/plans/strict_inspect_and_sort_gpu.json
```

## Two computers: Isaac Sim on a GPU box, the rest elsewhere

The Isaac step is a batch replay: a planned trajectory in, a report out. So the
work splits cleanly by files, the way Kaggle is used, with SSH as the transport.

| | main computer | GPU box |
|---|---|---|
| does | teach, plan (CPU, or cuRobo), MuJoCo, locating (MuJoCo images + detector), analysis | `sim_workflow.py --backend isaac`, RTX images, Isaac tests |
| needs | the full venv (`tools/setup_local.sh`, `SKIP_ISAAC=1` if it has no Isaac) | Linux, NVIDIA RTX GPU, the Isaac venv only (`ISAAC_ONLY=1`: Isaac Sim + this repository's core; no EAIK, coal, MuJoCo, detector) |
| sends | the repository's files (code, layouts, assets, teach tables) and the plan | the reports (`.json`, `.npz`, `.usda`, images) |

`tools/isaac/remote_isaac.sh` does the round trip over SSH with rsync:
1. sync this checkout's files (tracked, plus untracked files that are not
   ignored, with a REVISION file naming the commit);
2. send the git-ignored inputs you name (the plan);
3. run the command in the Isaac venv at the remote repository root;
4. bring back what you name, even when the command fails.

It needs `rsync` on both computers and key-based SSH from the main computer.

One-time setup:

```bash
# on the main computer: SSH key login to the GPU box (an ~/.ssh/config alias keeps it short)
ssh-copy-id user@gpu-box
export RSF_ISAAC_HOST=user@gpu-box          # also: RSF_ISAAC_DIR (default rsf), RSF_ISAAC_VENV (default venvs/rsf-isaac)
tools/isaac/remote_isaac.sh sync            # the repository -> ~/rsf on the GPU box

# on the GPU box (needs python3.12, python3.12-venv, rsync): the Isaac venv
ISAAC_ONLY=1 bash ~/rsf/tools/setup_local.sh ~/venvs/rsf-isaac

# on the main computer: SSH, the remote venv, isaacsim, the GPU
tools/isaac/remote_isaac.sh check
```

Each run: plan here, simulate there. Full plans are git-ignored
(`cells/*/plans/*.json`), so a plan exists only on the computer that made it:
make it first, then `--send` it.

The round trip can be tried first without planning, using the first strict
cycle, which is in git (two transfers, about a minute in Isaac):

```bash
tools/isaac/remote_isaac.sh run --collect reports/isaac --cmd "python tools/sim_workflow.py \
    cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --plan cells/tm_tray/plans/strict_cycle1.json.gz --backend isaac --transfers 2 --out reports/isaac/cycle1.json"
```

The full run:

```bash
L=cells/tm_tray/layouts/strict.yaml
WF=cells/tm_tray/workflows/inspect_and_sort.yaml
python tools/plan_workflow.py assemble $L $WF --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
    --paths cpu --out cells/tm_tray/plans/strict_inspect_and_sort_cpu.json
tools/isaac/remote_isaac.sh run --send cells/tm_tray/plans/strict_inspect_and_sort_cpu.json --collect reports/isaac \
    --cmd "python tools/sim_workflow.py $L $WF --plan cells/tm_tray/plans/strict_inspect_and_sort_cpu.json \
           --backend isaac --usd reports/isaac/strict.usda --report reports/isaac/run.npz --out reports/isaac/sim_strict.json"
```

The report (`reports/isaac/sim_strict.json`, the raw run in `run.npz`) is then
here, to compare with MuJoCo's. Images work the same way: `--cmd "python
tools/isaac/render_probe.py --usd reports/isaac/strict.usda --png
reports/isaac/strict.png"`, `--collect reports/isaac`.

Tested here with a stand-in for ssh that runs the remote side locally:
sync (214 files), send the plan, run one transfer (MuJoCo, as no Isaac is
installed here), fetch the report; and `ISAAC_ONLY=1` (the replay and judge
modules load without EAIK, coal or MuJoCo). Not tested over a real network.

To watch Isaac Sim on the GPU box from the main computer, Isaac Sim streams its
window (`isaacsim isaacsim.exp.full.streaming --no-window` on the GPU box, NVIDIA's
WebRTC streaming client here). That is for looking only: the replay itself stays
a batch job.

A live coupling, with the two computers exchanging state every step, is not
needed for anything done so far. It would be for locating during a run with
Isaac's camera images, or for driving the real controller against the simulation;
it would mean an Isaac-side service (ZeroMQ or gRPC, or ROS 2 through Isaac's
bridge) in place of the batch call.

## Isaac Sim's own Python (standalone package, air-gapped)

With the standalone Isaac Sim package there is no venv: its bundled interpreter,
`python.sh` (Windows: `python.bat`), is the Python. This assumes every
dependency is already installed in it:
* the Isaac side: pyyaml, trimesh, pillow, pytest;
* the main side, in addition: mujoco, coal, EAIK, motion_planner, urchin, the
  detector.

Replace `python` with it. `python.sh` sets its own `PYTHONPATH`, so the
repository must not rely on it. The tools put `src` on the path themselves, and
`tools/rsf_cli.py` stands in for `python -m rsf`.

```bash
ISAAC_PY=~/isaacsim/python.sh            # where the standalone package is
export OMNI_KIT_ACCEPT_EULA=YES MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd robot_simulation_framework

$ISAAC_PY -m pytest -q
$ISAAC_PY tools/rsf_cli.py teach cells/tm_tray/layouts/strict.yaml \
    --workflow cells/tm_tray/workflows/inspect_and_sort.yaml --out cells/tm_tray/teach/strict_inspect_and_sort
$ISAAC_PY tools/plan_workflow.py assemble cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --teach cells/tm_tray/teach/strict_inspect_and_sort.json --paths cpu --out cells/tm_tray/plans/strict_inspect_and_sort_cpu.json
$ISAAC_PY tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --plan cells/tm_tray/plans/strict_inspect_and_sort_cpu.json --backend isaac --usd reports/isaac/strict.usda \
    --out reports/isaac/sim_strict.json
```

Two computers: name the interpreter instead of a venv. The path is relative to
the remote home, or absolute. In `--cmd`, `python` then runs it:

```bash
export RSF_ISAAC_HOST=user@gpu-box RSF_ISAAC_PYTHON=isaacsim/python.sh
tools/isaac/remote_isaac.sh check            # prints the interpreter and where isaacsim comes from
tools/isaac/remote_isaac.sh run --send cells/tm_tray/plans/strict_inspect_and_sort_cpu.json --collect reports/isaac \
    --cmd "python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
           --plan cells/tm_tray/plans/strict_inspect_and_sort_cpu.json --backend isaac --out reports/isaac/sim_strict.json"
```

`remote_isaac.sh` only needs SSH and rsync between the two computers (no
internet). The main computer can use any Python 3.10+ that has the main
side's packages, or the same `python.sh` if Isaac Sim is installed there too.

Tested with a stand-in `python.sh` (one that replaces `PYTHONPATH`, then runs a
Python 3.12) through `remote_isaac.sh`, with no venv:
* `tools/rsf_cli.py`, pytest and a MuJoCo transfer ran;
* `check` found the interpreter;
* `check` does not mistake the `~/isaacsim` folder for the package.

Not tested with the real standalone package.

## What is Kaggle-only and not needed locally

`tools/isaac/kaggle_*.sh`, `vulkan_setup.sh` and the `isaac-sim-kaggle` skill:
Kaggle's GPUs come without Vulkan and need a cached environment. A desktop
NVIDIA driver includes Vulkan, and the venv is installed once.

Sources: [Isaac Sim Python installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html),
[isaacsim on PyPI](https://pypi.org/project/isaacsim/).

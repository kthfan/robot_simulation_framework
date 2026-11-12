# cuRobo on CPU: porting the CUDA kernels to Warp — effort estimate

Evaluated 2026-11-12 on kthfan/curobo `dev/main` (`445ed42`), on this CPU-only
container (warp-lang 1.19.0, torch 2.x CPU). Purpose: decide whether offline
teach-point / path generation (no GPU, PLAN §0) can use cuRobo instead of a
separate CPU stack.

## 1. What runs on the GPU today

cuRobo's Python package launches two kinds of kernels:

| Family | Where | Size | CPU today |
|---|---|---|---|
| **CUDA C++** (`.cuh`, JIT-compiled through `cuda.core`/NVRTC) | `curobo/_src/curobolib/kernels/`, dispatched by `curobolib/backends/cuda_core_backend/*` | 53 `__global__` kernels, ~13.5k lines (+1.5k `helper_math.h`), 31 `launch_*` entry points in 9 modules | **No** |
| **Warp** (`@wp.kernel`, ~145 `wp.launch`) | scene collision (primitives, mesh, voxel, swept), robot-mesh collision/self-collision, tool-pose / cspace / bound costs, RRT, perception | ~26k lines of Python incl. kernels | **Mostly yes** (see §2) |

The 8 `.cu` files are in `cpp_runtime/` (C++ deployment export) and do not
concern the Python package.

The CUDA modules sit behind a clean seam: `curobolib/backends/__init__.py`
lazily proxies `kinematics`, `geometry`, `dynamics`, `optimization`, `pba`,
`trajectory`, `trajectory_seed_fitter` to `cuda_core_backend`. A
`warp_backend` (or `torch_backend`) can be added next to it and chosen by
tensor device, without touching the callers (`cuda_ops/*` autograd Functions).

### CUDA kernels by module

| Module | Kernels / launches | Lines (approx.) | GPU-specific constructs | Needed offline? |
|---|---|---|---|---|
| kinematics (FK, spheres, Jacobian, backward) | 5 / 5 | 1,700 | warp shuffles, shared memory | **yes** (everything) |
| geometry: self-collision (sphere pairs) | 2 / 2 | 1,200 | shared-memory tiling, ~30 sync/shuffle sites | **yes** |
| optimization: L-BFGS step, compact step, line search | 3 / 3 | 1,800 | block reductions, ~36 sync sites | yes for IK / TrajOpt |
| trajectory: B-spline interp. fwd/bwd, dt scaling, legacy diff/integration | 8 / 7 | 2,600 | few | yes for TrajOpt |
| trajectory_timeopt (time-optimal retiming) | 16 / – | 1,200 | ~36 sync sites | yes for final timing |
| trajectory_seed_fitter | 7 / 6 | 400 | some | yes for seeds |
| dynamics (RNEA, CRBA fwd/bwd) | 4 / 4 | 2,600 | shared memory | no (inverse dynamics can be turned off) |
| pose_refiner (LM) | 5 / 5 | 330 | | no (perception) |
| pba (3-D EDT for ESDF) | 3 / 1 | 450 | | no (perception) |

## 2. Experiments on this CPU

1. **Warp itself on CPU**: tile kernels (`wp.tile_load/tile_sum/tile_atomic_add`
   via `launch_tiled`), `wp.Mesh` + `mesh_query_point`, and `wp.from_torch` on
   CPU tensors all work (Warp 1.19.0).
2. **cuRobo kinematics on CPU**: fails immediately. `DeviceCfg()` defaults to
   `cuda:0` at 109 call sites (default arguments), and the FK path calls the
   CUDA kernel (`kinematics_loader.py:1050` raises "q is not on cuda").
3. **cuRobo scene collision (Warp) on CPU**, with the default device switched to
   CPU and one shim (`torch.cuda.get_device_properties().multi_processor_count`,
   used only for launch sizing in `wp_autograd.py:138`):
   - cuboid obstacle: **correct** (sphere 5 cm inside → cost 0.055; clear → 0).
   - mesh obstacle: a sphere whose centre is inside the mesh is correct
     (0.155), but a sphere whose centre is **outside while its surface
     penetrates** returns **0**. Either a CPU-specific defect or the same
     behaviour on GPU; it has to be compared on Kaggle before anything is
     built on it. Silent wrong answers like this are the main risk of a port,
     so every kernel needs a CPU-vs-GPU parity test.
4. Other CUDA couplings in Python: 96 `torch.cuda.*` uses (34 `current_stream`,
   17 `Event`, 12 `Stream`, 11 `get_device_properties`, CUDA graphs), all
   replaceable by no-ops or guards on CPU.

## 3. Effort (one engineer, focused; includes parity tests on Kaggle)

| Step | Scope | Estimate | Unlocks |
|---|---|---|---|
| A. Device plumbing | `DeviceCfg` default, `torch.cuda` guards, CUDA graphs/streams off on CPU, launch sizing shims | 3–5 days | Warp paths on CPU |
| B. Kinematics | 5 launches → Warp (or torch, with autograd for backward) | 5–8 days | FK, spheres, Jacobian on CPU |
| C. Self-collision | 2 kernels → Warp | 4–6 days | `RobotCollisionChecker` incl. exact mesh check (Warp) on CPU |
| D. Optimizer | L-BFGS, compact step, line search | 6–9 days | IK, TrajOpt |
| E. Trajectory | B-spline, legacy diff/integration, seed fitter | 8–12 days | TrajOpt, MotionPlanner |
| F. Time-optimal retiming | 16 kernels | 5–8 days | final trajectory timing |
| G. Parity harness | per-kernel CPU vs GPU comparison on Kaggle, fuzzed shapes | 3–5 days (shared across all) | trust |
| — Dynamics, pose refiner, PBA | not needed offline | 0 (skip) | |

- **A + B + C + G ≈ 3–4 weeks** → cuRobo's collision checking (spheres, scene,
  exact robot-mesh, mesh self-collision, attachments) and FK on CPU. This is
  the part the offline pipeline needs most: the same collision verdicts as the
  GPU side.
- **+ D + E + F ≈ another 4–6 weeks** → IK / TrajOpt / `MotionPlanner` on CPU.
- Writing B/D/E as **plain PyTorch** instead of Warp is roughly 30–40 % less
  work (autograd replaces hand-written backward kernels) but slower and more
  memory-hungry; acceptable offline.

Uncertainty: medium. The numbers assume the Warp paths need only plumbing
(experiment 3 says mostly, with one open defect) and that the CUDA kernels'
algorithms port 1:1. Shared-memory reductions become serial loops on CPU,
which is simple to write but makes per-kernel speed hard to predict.

### Expected CPU speed

cuRobo's solvers win on the GPU by running many seeds in parallel (IK 32+
seeds, TrajOpt particles). On 4–16 CPU cores, expect roughly 10–100× slower
than a T4 per call: collision checks of thousands of configurations stay in
milliseconds-to-tens-of-milliseconds; a full `plan_pose` likely seconds.
Fine for offline generation, not for real-time.

## 4. Recommendation

1. **Do A + B + C + G first** (collision + FK on CPU). It removes the risk of two
   collision implementations disagreeing (offline CPU vs real-time GPU), which
   is exactly the failure class AI-Robotics-V2 D22 warns about. Until then,
   the framework's CPU collision backend is used and cross-checked against
   cuRobo on Kaggle.
2. Decide on D–F after measuring A–C's CPU speed. Meanwhile offline planning
   keeps EAIK + RRT-Connect on the CPU backend.
3. The work belongs in kthfan/curobo (a `warp_backend` beside
   `cuda_core_backend`), not in this repo. This session can only write to
   robot_simulation_framework; adding kthfan/curobo to the session is needed.

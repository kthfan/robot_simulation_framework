#!/usr/bin/env bash
# Local setup (Linux x86-64, NVIDIA RTX GPU): one Python 3.12 venv with Isaac Sim 6.2 and
# everything this repository needs. See docs/local_setup.md.
#
#   bash tools/setup_local.sh [VENV_DIR]          # default: ~/venvs/rsf-isaac
#
# Environment:
#   TORCH_CUDA=cu128|cu130    PyTorch build (default cu128: CUDA 12)
#   ISAAC_VERSION=6.2.0.0     Isaac Sim version
#   DETECTOR_DIR=...          kthfan/aprilgriddetector checkout (default: ../kthfan/aprilgriddetector,
#                             cloned if missing; needs Rust: https://rustup.rs)
#   SKIP_ISAAC=1              skip PyTorch + Isaac Sim (tests the rest only; MuJoCo/CPU work)
#
# Never install usd-core or opencv-python(-headless) into this venv: Isaac Sim brings its
# own pxr (usd-exchange) and cv2 (opencv-python-headless-noffmpeg), and pins numpy and mujoco.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${1:-$HOME/venvs/rsf-isaac}
TORCH_CUDA=${TORCH_CUDA:-cu128}
ISAAC_VERSION=${ISAAC_VERSION:-6.2.0.0}
DETECTOR_DIR=${DETECTOR_DIR:-$ROOT/../kthfan/aprilgriddetector}
say() { printf '\n== %s\n' "$*"; }

say "checks"
command -v python3.12 >/dev/null || { echo "python3.12 not found (Ubuntu: sudo apt install python3.12 python3.12-venv python3.12-dev)"; exit 1; }
ldd --version | head -1
command -v g++ >/dev/null || { echo "g++ not found (EAIK builds C++17): sudo apt install build-essential"; exit 1; }
command -v git >/dev/null || { echo "git not found"; exit 1; }
if [ "${SKIP_ISAAC:-0}" != 1 ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no NVIDIA GPU visible"; exit 1; }
fi

say "venv $VENV (Python 3.12)"
[ -x "$VENV/bin/python" ] || python3.12 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

if [ "${SKIP_ISAAC:-0}" != 1 ]; then
  say "PyTorch 2.11.0 ($TORCH_CUDA) and Isaac Sim $ISAAC_VERSION (~25 GB)"
  pip install torch==2.11.0 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
  pip install "isaacsim[all,extscache]==$ISAAC_VERSION" --extra-index-url https://pypi.nvidia.com
fi

if [ "${SKIP_ISAAC:-0}" = 1 ]; then
  pip install -q opencv-python-headless       # with Isaac Sim, its own OpenCV provides cv2
fi

say "this repository (editable) and its dependencies"
# mesh, collision (coal), eaik (kthfan/EAIK, built from source: C++17), mujoco
pip install -e "$ROOT[mesh,collision,eaik,mujoco]" pytest maturin

say "AprilGrid detector (kthfan/aprilgriddetector, Rust)"
if ! command -v cargo >/dev/null; then
  echo "cargo not found: install Rust (https://rustup.rs), then rerun; skipping the detector (locating tools need it)"
else
  [ -d "$DETECTOR_DIR" ] || git clone https://github.com/kthfan/aprilgriddetector "$DETECTOR_DIR"
  (cd "$DETECTOR_DIR" && maturin build --release -i "$VENV/bin/python" --out dist \
      --manifest-path crates/aprilgrid-py/Cargo.toml)
  pip install --force-reinstall --no-deps "$DETECTOR_DIR"/dist/aprilgrid_detector-*-cp312-*.whl
fi

say "versions"
python - <<'PY'
import importlib
for m in ("numpy", "mujoco", "coal", "trimesh", "eaik", "cv2", "aprilgrid_detector", "torch", "isaacsim"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:20s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:20s} MISSING ({type(e).__name__})")
PY
say "done. Next: source $VENV/bin/activate; see docs/local_setup.md (tests, then the workflow)"

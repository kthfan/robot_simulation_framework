#!/usr/bin/env bash
# Kaggle (inside the runner's Python 3.12 venv), when the cache is built: Isaac Sim and a few
# light packages. ISAAC_VERSION and ISAAC_EXTRA_PIP come from isaac_kaggle.sh rebuild.
# Never add usd-core here: it ships its own pxr, which conflicts with Isaac's.
set -euo pipefail
pip install -q "isaacsim[all,extscache]==${ISAAC_VERSION:-6.2.0.0}" --extra-index-url https://pypi.nvidia.com 2>&1 | tail -3
pip install -q ${ISAAC_EXTRA_PIP:-numpy pyyaml trimesh pillow pytest zstandard} 2>&1 | tail -3

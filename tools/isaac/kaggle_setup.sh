#!/usr/bin/env bash
# Kaggle setup for tools/isaac/kaggle_check.sh: Isaac Sim 6.2 (Python 3.12) and the framework's light deps.
set -euo pipefail
pip install -q 'isaacsim[all,extscache]==6.2.0.0' --extra-index-url https://pypi.nvidia.com 2>&1 | tail -3
pip install -q numpy pyyaml trimesh pillow pytest 2>&1 | tail -3

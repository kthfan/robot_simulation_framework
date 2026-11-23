#!/usr/bin/env bash
# Package the standalone kaggle-isaac-sim skill: tools/isaac/standalone/kaggle-isaac-sim.zip
# (unzip into ~/.claude/skills/, or upload it as a skill).
set -euo pipefail
cd "$(dirname "$0")"
rm -f kaggle-isaac-sim.zip
find kaggle-isaac-sim -name __pycache__ -prune -exec rm -rf {} +
zip -qrX kaggle-isaac-sim.zip kaggle-isaac-sim
unzip -l kaggle-isaac-sim.zip

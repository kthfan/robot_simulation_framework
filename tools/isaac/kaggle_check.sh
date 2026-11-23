#!/usr/bin/env bash
# Isaac Sim on a Kaggle T4 (P5): probe, load the compiled strict cell, replay its first cycle.
#   bash kaggle_run.sh run --python 3.12 --setup "bash tools/isaac/kaggle_setup.sh" --cmd "bash tools/isaac/kaggle_check.sh"
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src
mkdir -p reports/isaac
rc=0
timeout 600 python tools/isaac/probe.py --out reports/isaac/probe_box.json || rc=1
timeout 600 python - <<'PY' || rc=1
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from rsf.model.layout import load_world
from rsf.compile.usd import world_stage
world_stage(load_world("cells/tm_tray/layouts/strict.yaml"), "reports/isaac/strict.usda")
print("compiled reports/isaac/strict.usda")
app.close()
PY
timeout 900 python -m pytest -q -p no:cacheprovider tests/test_isaac.py --junitxml reports/isaac/test_isaac.xml || rc=1
timeout 600 python tools/isaac/probe.py --usd reports/isaac/strict.usda --out reports/isaac/probe_strict.json || rc=1
timeout ${ISAAC_SIM_TIMEOUT:-2400} python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml \
    cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_cycle1.json.gz \
    --backend isaac --transfers ${ISAAC_TRANSFERS:-2} --usd reports/isaac/strict_run.usda \
    --report reports/isaac/sim_strict_report.npz --out reports/isaac/sim_strict.json || rc=1
gzip -kf reports/isaac/strict.usda 2>/dev/null; rm -f reports/isaac/strict.usda reports/isaac/strict_run.usda
exit $rc

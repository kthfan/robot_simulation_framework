#!/usr/bin/env bash
# The example cell's whole pipeline, stage by stage (docs/new_cell.md). From the repo root:
#   bash cells/example_pick_place/run.sh            # ~20 s on a laptop CPU, no GPU
# Each stage writes a file the next one reads; each prints a verdict and exits non-zero on failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
C=cells/example_pick_place
L=$C/layouts/main.yaml
W=$C/workflows/move_blocks.yaml

python -m rsf audit  "$L"                                                   # 1. the layout is sound
python -m rsf expand "$L" "$W"                                              # 2. what the run does, in order
python -m rsf teach  "$L" --workflow "$W" --out $C/teach/move_blocks         # 3. joints for every waypoint
python tools/plan_workflow.py assemble "$L" "$W" --teach $C/teach/move_blocks.json \
    --paths cpu --out $C/plans/move_blocks.json                             # 4. one timed, checked trajectory
python tools/sim_workflow.py "$L" "$W" --plan $C/plans/move_blocks.json \
    --out reports/example/sim_move_blocks.json "$@"                         # 5. physics + judge
# extra arguments go to the simulation, e.g. --live 8765 to watch it in a browser

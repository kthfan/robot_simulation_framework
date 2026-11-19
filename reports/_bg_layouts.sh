#!/bin/bash
# P4 for as_built / recommended: workflow teach, then a CPU-planned, gated full trajectory
cd "$(dirname "$0")/.."
L=$1
PYTHONPATH=src python3 -m rsf teach cells/tm_tray/layouts/$L.yaml --workflow cells/tm_tray/workflows/inspect_and_sort.yaml \
    --out cells/tm_tray/teach/${L}_inspect_and_sort > reports/_teach_wf_$L.log 2>&1 || { echo TEACH_FAILED > reports/_bg_$L.done; exit 1; }
PYTHONPATH=src python3 tools/plan_workflow.py assemble cells/tm_tray/layouts/$L.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --teach cells/tm_tray/teach/${L}_inspect_and_sort.json --paths cpu --out cells/tm_tray/plans/${L}_inspect_and_sort_cpu.json \
    > reports/_plan_full_cpu_$L.log 2>&1
echo "exit $?" > reports/_bg_$L.done

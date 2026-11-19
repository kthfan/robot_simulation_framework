#!/bin/bash
# P4 for one layout, unattended: workflow teach (skipped if the table exists),
# then a CPU-planned, gated full trajectory. Logs and a .done marker in reports/.
#   nohup tools/plan_layout_bg.sh as_built > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
L=$1
mkdir -p reports
T=cells/tm_tray/teach/${L}_inspect_and_sort
if [ ! -f $T.json ]; then
  PYTHONPATH=src python3 -m rsf teach cells/tm_tray/layouts/$L.yaml --workflow cells/tm_tray/workflows/inspect_and_sort.yaml \
      --out $T > reports/_teach_wf_$L.log 2>&1 || { echo TEACH_FAILED > reports/_bg_$L.done; exit 1; }
fi
PYTHONPATH=src python3 tools/plan_workflow.py assemble cells/tm_tray/layouts/$L.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --teach $T.json --paths cpu --out cells/tm_tray/plans/${L}_inspect_and_sort_cpu.json \
    > reports/_plan_full_cpu_$L.log 2>&1
echo "exit $?" > reports/_bg_$L.done

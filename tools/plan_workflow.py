"""P4: plan a workflow run into one timed, checked joint trajectory.

    # 1. the free moves as a cuRobo job (commit it; Kaggle clones the repo)
    PYTHONPATH=src python tools/plan_workflow.py job cells/tm_tray/layouts/strict.yaml \
        cells/tm_tray/workflows/inspect_and_sort.yaml --teach cells/tm_tray/teach/strict_inspect_and_sort.json \
        --cycles 1 --out tools/kaggle/jobs/strict_1cycle.json
    # 2. on a GPU: python tools/kaggle/curobo_free_plan.py --job ... --out results/....json
    # 3. time everything, check every sample on the CPU, write the trajectory
    PYTHONPATH=src python tools/plan_workflow.py assemble <layout> <workflow> --teach ... --cycles 1 \
        --paths results/....json --out cells/tm_tray/plans/strict_1cycle.json

``--paths cpu`` plans the free moves on the CPU (RRT-Connect + shortcut, the
fallback without a GPU); ``--paths linear`` uses straight joint lines (tests;
the CPU check still decides).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsf.model.layout import load_world  # noqa: E402


def load(a):
    from rsf.model.workflow import Workflow
    from rsf.planning import plan as PL
    world = load_world(a.layout)
    wf = Workflow.from_file(a.workflow)
    if a.cycles:
        wf.max_cycles = a.cycles
    doc = json.loads(Path(a.teach).read_text())
    teach = {s["sequence"]: s["points"] for s in doc["sequences"]}
    return world, wf, PL.segments_for_workflow(world, wf, teach)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["job", "assemble"])
    ap.add_argument("layout")
    ap.add_argument("workflow")
    ap.add_argument("--teach", required=True, help="teach table JSON covering the run's sequences")
    ap.add_argument("--cycles", type=int, default=0)
    ap.add_argument("--paths", help="assemble: cuRobo results JSON, 'cpu' or 'linear'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from rsf.planning import plan as PL
    world, wf, segs = load(a)
    kinds = {k: sum(s.kind == k for s in segs) for k in ("line", "path", "free", "hold")}
    print(f"{len(segs)} segments {kinds}")
    if a.cmd == "job":
        PL.write_job(world, segs, a.out, root=ROOT, meta={"layout": a.layout, "workflow": a.workflow,
                                                           "cycles": a.cycles, "teach": a.teach})
        print(f"job -> {a.out}")
        return 0
    from rsf.planning.teach import make_engine
    eng = make_engine(world)
    if a.paths == "linear":
        for s in segs:
            if s.kind == "free":
                s.Q = np.linspace(s.q_start, s.q_goal, 50)
    elif a.paths == "cpu":                                    # no GPU: RRT-Connect on the CPU
        for s in segs:
            if s.kind == "free":
                r = PL.plan_free_cpu(eng.arm, eng.checker, s)
                print(f"  {'ok  ' if r['success'] else 'FAIL'} {s.name}: {r}", flush=True)
    else:
        bad = PL.attach_results(segs, json.loads(Path(a.paths).read_text()))
        if bad:
            print(f"no path for {len(bad)} free segments: {bad[:5]}")
    t0 = time.time()
    traj, problems = PL.assemble(world, eng.arm, eng.checker, segs)
    traj.meta.update({"workflow": a.workflow, "cycles": a.cycles, "teach": a.teach, "paths": a.paths,
                      "problems": problems, "ok": not problems, "seconds": round(time.time() - t0, 1)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(traj.to_dict()))
    summary = {"schema": "rsf.trajectory_summary/1", "duration": round(float(traj.t[-1]), 3),
               "samples": len(traj.t), "segments": traj.segments, "meta": traj.meta}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(f"trajectory {traj.t[-1]:.1f} s, {len(traj.t)} samples, {len(problems)} problems -> {a.out}")
    for p in problems[:15]:
        print("  PROBLEM", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

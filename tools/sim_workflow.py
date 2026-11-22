"""P5: replay a planned workflow run in simulation and judge every transfer (D30 criteria).

    PYTHONPATH=src python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml \\
        cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_inspect_and_sort_gpu.json \\
        --transfers 2 --out reports/sim_strict_1cycle.json

``--transfers N`` simulates up to the end of the N-th transfer (0: the whole
plan). ``--set friction.robot=0.3`` etc. changes simulation parameters
(``rsf.sim.common.SimParams``; Isaac: ``rsf.sim.isaac_cell.IsaacParams``) for
sensitivity runs. ``--backend isaac`` replays it in Isaac Sim (PhysX) instead of
MuJoCo: same world, servo and judge (needs Isaac Sim, see ``docs/p5_isaac.md``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def params(sets, backend="mujoco"):
    if backend == "isaac":
        from rsf.sim.isaac_cell import IsaacParams as P
    else:
        from rsf.sim.common import SimParams as P
    p = P()
    for s in sets or []:
        k, v = s.split("=", 1)
        v = json.loads(v)
        if "." in k:
            a, b = k.split(".", 1)
            getattr(p, a)[b] = v
        else:
            setattr(p, k, tuple(v) if isinstance(v, list) else v)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("workflow")
    ap.add_argument("--plan", required=True, help="rsf.trajectory/1 JSON (tools/plan_workflow.py assemble)")
    ap.add_argument("--transfers", type=int, default=0)
    ap.add_argument("--set", action="append", help="SimParams override, key=json")
    ap.add_argument("--backend", choices=("mujoco", "isaac"), default="mujoco")
    ap.add_argument("--usd", help="isaac: where to write the compiled stage (default: a temporary file)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from rsf.model.layout import load_world
    from rsf.model.workflow import Workflow
    from rsf.sim import judge as J
    world = load_world(a.layout)
    wf = Workflow.from_file(a.workflow)
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    traj = json.loads(Path(a.plan).read_text())
    t_end = None
    if a.transfers:
        trs = J.transfers(world, wf, traj["segments"], occ)
        t_end = trs[min(a.transfers, len(trs)) - 1].t1 + 0.5
    if a.backend == "isaac":
        from rsf.sim.isaac_cell import IsaacCell
        cell = IsaacCell(world, occ, params(a.set, "isaac"), usd_path=a.usd)
    else:
        from rsf.sim.mujoco_cell import MujocoCell
        cell = MujocoCell(world, occ, params(a.set))
    t0 = time.time()
    rep = cell.run(traj, t_end=t_end, progress=lambda t, w: print(f"  sim t={t:7.1f} s  wall {w:6.1f} s", flush=True))
    res = J.judge(world, wf, rep, occ, traj=traj)
    res["meta"] = {"layout": a.layout, "workflow": a.workflow, "plan": a.plan, "sim_seconds": round(float(rep.t[-1]), 2),
                   "wall_seconds": round(time.time() - t0, 1), "backend": a.backend, "params": {k: v for k, v in vars(cell.p).items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for t in res["transfers"]:
        ch = " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in t["checks"].items())
        print(f"{'PASS' if t['ok'] else 'FAIL'} {t['src']} -> {t['dst']}: seat {t['seat_mm']:.2f} mm / {t['seat_deg']:.2f} deg, "
              f"slip {t['slip_mm']:.2f} mm, track {t['track_mm']:.2f} mm, carry tilt {t['carry_tilt_deg']:.1f} deg  [{ch}]")
        for c in t["contacts"][:5]:
            print(f"     contact {c['link']} vs {c['other']}: {c['n']}x, {c['depth_mm']} mm, t {c['t_first']}..{c['t_last']}")
    for c in res["other_contacts"][:5]:
        print(f"  other contact {c['link']} vs {c['other']}: {c['n']}x, {c['depth_mm']} mm, t {c['t_first']}..{c['t_last']}")
    n = sum(t["ok"] for t in res["transfers"])
    print(f"SIM {'PASS' if res['ok'] else 'FAIL'}: {n}/{len(res['transfers'])} transfers "
          f"({res['meta']['sim_seconds']} s simulated, {res['meta']['wall_seconds']} s wall) -> {a.out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

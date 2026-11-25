"""P5: replay a planned workflow run in simulation and judge every transfer (D30 criteria).

    PYTHONPATH=src python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml \\
        cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_inspect_and_sort_gpu.json \\
        --transfers 2 --out reports/sim_strict_1cycle.json

``--transfers N`` simulates up to the end of the N-th transfer (0: the whole
plan). ``--set friction.robot=0.3`` etc. changes simulation parameters
(``rsf.sim.common.SimParams``; Isaac: ``rsf.sim.isaac_cell.IsaacParams``) for
sensitivity runs. ``--backend isaac`` replays it in Isaac Sim (PhysX) instead of
MuJoCo: same world, servo and judge (needs Isaac Sim, see ``docs/p5_isaac.md``).

``--live [HOST:]PORT`` serves a live view while it runs: the simulator renders the
views (Isaac: RTX), a browser shows them at ``http://HOST:PORT/`` and programs read
``/api/v1/...`` (``rsf.live``, ``docs/live_view.md``). ``--live 8765`` listens on this
computer only (reach it over SSH: ``tools/isaac/remote_isaac.sh run --view 8765``);
``--live 0.0.0.0:8765`` on the network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_plan(path) -> dict:
    """A ``rsf.trajectory/1`` JSON file, gzipped (``.gz``) or not."""
    import gzip
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_text())


def save_report(rep, path) -> None:
    """An ``ExecReport`` as ``.npz`` (arrays; contacts, segments and meta as JSON)."""
    import numpy as np
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=rep.t, flange=rep.flange, flange_planned=rep.flange_planned, q=rep.q,
                        object_names=np.array(list(rep.objects)), objects=np.array(list(rep.objects.values())),
                        json=json.dumps({"contacts": rep.contacts, "segments": rep.segments, "meta": rep.meta},
                                        default=str))


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


def start_live(a, world, cell, traj, t_end):
    """The live server and the capture hook of a run (``--live``)."""
    from rsf.live.capture import LiveCapture
    from rsf.live.hub import Hub
    from rsf.live.server import LiveServer, parse_address
    from rsf.sim.common import View, overview
    hub = Hub()
    host, port = parse_address(a.live)
    server = LiveServer(hub, host, port).start()
    views = []
    for spec in a.live_view or ["overview"]:
        if spec == "overview":
            views.append(overview(world, width=a.live_size[0], height=a.live_size[1]))
        else:
            d = json.loads(spec)
            d.setdefault("width", a.live_size[0])
            d.setdefault("height", a.live_size[1])
            views.append(View.from_spec(d))
    T = traj["t"]
    hub.set_info(backend=a.backend, layout=a.layout, workflow=a.workflow, plan=a.plan,
                 t_end=float(t_end if t_end is not None else T[-1]))
    cap = LiveCapture(hub, cell, views, fps=a.live_fps, pace=a.live_pace, segments=traj.get("segments", ()),
                      t_end=float(t_end if t_end is not None else T[-1]), quality=a.live_quality)
    print(f"[live] {server.url}  (views: {', '.join(v.name for v in views)}; bound to {host}:{server.address[1]})",
          flush=True)
    cap.warmup()
    return server, cap


def stop_live(a, server, cap, state="done", **result):
    cap.finish(state, **result)
    if a.live_hold:
        print(f"[live] serving {server.url} {'until Ctrl-C' if a.live_hold < 0 else f'for {a.live_hold:g} s'}", flush=True)
        try:
            time.sleep(a.live_hold if a.live_hold > 0 else 1e9)
        except KeyboardInterrupt:
            pass
    server.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("workflow")
    ap.add_argument("--plan", required=True, help="rsf.trajectory/1 JSON (tools/plan_workflow.py assemble)")
    ap.add_argument("--transfers", type=int, default=0)
    ap.add_argument("--set", action="append", help="SimParams override, key=json")
    ap.add_argument("--backend", choices=("mujoco", "isaac"), default="mujoco")
    ap.add_argument("--usd", help="isaac: where to write the compiled stage (default: a temporary file)")
    ap.add_argument("--report", help="also save the raw execution report (.npz)")
    ap.add_argument("--out", required=True)
    lv = ap.add_argument_group("live view (rsf.live)")
    lv.add_argument("--live", metavar="[HOST:]PORT", help="serve a live view (default host 127.0.0.1)")
    lv.add_argument("--live-view", action="append", metavar="SPEC",
                    help="a view: 'overview' (default) or JSON {name, eye, target, fov_deg, width, height}; repeatable")
    lv.add_argument("--live-fps", type=float, default=5.0, help="rendered frames per wall-clock second (default 5)")
    lv.add_argument("--live-size", type=int, nargs=2, metavar=("W", "H"), default=(960, 540))
    lv.add_argument("--live-pace", type=float, default=0.0,
                    help="no faster than this x real time (0: as fast as it runs)")
    lv.add_argument("--live-hold", type=float, default=0.0,
                    help="seconds to keep serving after the run (-1: until Ctrl-C)")
    lv.add_argument("--live-quality", type=int, default=80, help="JPEG quality")
    a = ap.parse_args(argv)
    if a.live and a.backend == "mujoco" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")         # offscreen; MUJOCO_GL=osmesa without a GPU
    from rsf.model.layout import load_world
    from rsf.model.workflow import Workflow
    from rsf.sim import judge as J
    world = load_world(a.layout)
    wf = Workflow.from_file(a.workflow)
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    traj = load_plan(a.plan)
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
    live = start_live(a, world, cell, traj, t_end) if a.live else None
    t0 = time.time()
    try:
        rep = cell.run(traj, t_end=t_end, on_sample=live[1] if live else None,
                       progress=lambda t, w: print(f"  sim t={t:7.1f} s  wall {w:6.1f} s", flush=True))
    except BaseException as e:
        if live:
            stop_live(a, *live, state="error", error=repr(e))
        raise
    if a.report:                                      # the raw run, kept even if judging fails
        save_report(rep, a.report)
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
    if live:
        stop_live(a, *live, state="done" if res["ok"] else "failed", ok=res["ok"], passed=n,
                  transfers=[{k: t[k] for k in ("src", "dst", "ok", "seat_mm", "slip_mm", "track_mm", "carry_tilt_deg")}
                             for t in res["transfers"]])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""P5: replay a planned workflow run in simulation and judge every transfer (D30 criteria).

    PYTHONPATH=src python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml \\
        cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_inspect_and_sort_gpu.json \\
        --transfers 2 --out reports/sim_strict_1cycle.json

``--transfers N`` simulates up to the end of the N-th transfer (0: the whole
plan). ``--set friction.robot=0.3`` etc. changes simulation parameters
(``rsf.sim.common.SimParams``; Isaac: ``rsf.sim.isaac_cell.IsaacParams``) for
sensitivity runs. ``--backend isaac`` replays it in Isaac Sim (PhysX) instead of
MuJoCo: same world, servo and judge (needs Isaac Sim, see ``docs/p5_isaac.md``).

**Reads.** A layout with ``process.reads`` (``rsf.sim.reads``) has its plan's marker
reads taken as the run passes them: the read camera renders (MuJoCo, or Isaac's RTX),
the detector finds the boards, and ``reads`` in the output says how far each is from
the simulated truth. ``--nominal LAYOUT`` (the layout the robot was taught in; default:
the simulated one) and ``--located OUT.yaml`` fit each located fixture's displacement
and write the corrected layout to teach and plan in; ``--read-images DIR`` keeps the
images; ``--no-reads`` skips them.

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


def start_debugpy(spec, wait: bool) -> None:
    """``--debugpy``: listen for a debugger. It has no password: anyone who reaches the
    port runs code here, so it listens on 127.0.0.1 unless told otherwise (SSH tunnel)."""
    import debugpy

    from rsf.live.server import parse_address
    host, port = parse_address(spec)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[debugpy] WARNING: listening on {host}: debugpy has no password, anyone who reaches "
              f"port {port} can run code as this user", flush=True)
    debugpy.listen((host, port))
    print(f"[debugpy] listening on {host}:{port} (attach from VS Code: Python Debugger, Remote Attach)", flush=True)
    if wait:
        print("[debugpy] waiting for the debugger ...", flush=True)
        debugpy.wait_for_client()


def start_live(a, world, wf, cell, traj, t_end):
    """The live server and the capture hook of a run (``--live``)."""
    import secrets

    import numpy as np

    from rsf.live import control, debug
    from rsf.live import record as live_record
    from rsf.live import views as live_views
    from rsf.live.capture import LiveCapture
    from rsf.live.hub import Hub
    from rsf.live.server import LiveServer, parse_address
    from rsf.sim.common import View, overview
    hub = Hub()
    host, port = parse_address(a.live)
    ctx = None
    if a.live_cert:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.live_cert, a.live_key)
    gated = a.live_control or a.live_debug
    token = (os.environ.get("RSF_LIVE_TOKEN") or secrets.token_urlsafe(24)) if gated else None
    server = LiveServer(hub, host, port, token=token, ssl_context=ctx)
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
    t_stop = float(t_end if t_end is not None else T[-1])
    hub.set_info(backend=a.backend, layout=a.layout, workflow=a.workflow, plan=a.plan, t_end=t_stop, pace=a.live_pace)
    ctl = control.Controller(hub) if gated else None
    console = None
    if ctl is not None:
        control.register(server, ctl)
    if a.live_debug:
        console = debug.DebugConsole(hub)
        debug.register(server, console)
    recorder = live_record.Recorder(hub, a.live_record_dir, meta={
        "layout": a.layout, "workflow": a.workflow, "plan": a.plan, "backend": a.backend})
    cap = LiveCapture(hub, cell, views, fps=a.live_fps, pace=a.live_pace, segments=traj.get("segments", ()),
                      t_end=t_stop, quality=a.live_quality, controller=ctl, console=console,
                      fps_interactive=a.live_fps_interactive, recorder=recorder)
    live_views.register(server, cap)
    live_record.register(server, recorder, cap)
    rtc_note = None
    if a.live_webrtc:
        from rsf.live import webrtc
        why = webrtc.missing()
        if why:
            rtc_note = f"off: {why}; the viewer uses MJPEG"
        else:
            rtc = webrtc.register(server, hub, ice_servers=a.live_webrtc_ice or (), codec=a.live_webrtc_codec,
                                  max_peers=a.live_webrtc_peers, max_kbps=a.live_webrtc_kbps or None)
            rtc_note = (f"on ({rtc.codec}, up to {rtc.max_peers} viewers; media over UDP: the browser must reach this "
                        f"computer{' or the ICE servers' if rtc.ice_servers else ''}, else it falls back to MJPEG)")
    if console is not None:
        console.ns.update(cell=cell, world=world, workflow=wf, traj=traj, hub=hub, capture=cap, np=np,
                          pause=lambda: ctl.command("pause"), resume=lambda: ctl.command("resume"),
                          step=lambda dt=0.1: ctl.command("step", dt=dt))
    server.start()
    print(f"[live] {server.url}  (views: {', '.join(v.name for v in views)}; bound to {host}:{server.address[1]})",
          flush=True)
    print(f"[live] recordings: {recorder.dir.resolve()} ({'ffmpeg ' + recorder.codec[1] if recorder.available else 'off: no ffmpeg'})",
          flush=True)
    if rtc_note:
        print(f"[live] WebRTC {rtc_note}", flush=True)
    if token:
        what = "control and a Python console" if a.live_debug else "control"
        hint = "" if os.environ.get("RSF_LIVE_TOKEN") else f" token {token}"
        print(f"[live] {what} enabled:{hint}  (open {server.url}#token=<token>; RSF_LIVE_TOKEN sets it)", flush=True)
        if host not in ("127.0.0.1", "localhost", "::1") and ctx is None:
            print("[live] WARNING: the token crosses the network in clear text (HTTP): use --live-cert/--live-key "
                  "or an SSH tunnel on networks you do not trust", flush=True)
        if a.live_debug and host not in ("127.0.0.1", "localhost", "::1"):
            print(f"[live] WARNING: anyone on the network with the token can run code as this user on {host}",
                  flush=True)
    cap.warmup()
    return server, cap


def interrupt_on_hangup() -> None:
    """Ctrl-C, SIGTERM and SIGHUP (a closed terminal or SSH session) end a live run: the
    server closes and the port is free, instead of the process serving on unseen. (SIGINT
    too: Isaac's app may have replaced Python's own handler.)"""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)
    for sig in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), handler)


def stop_live(a, server, cap, state="done", hold=True, **result):
    try:
        cap.finish(state, **result)
    except Exception as e:  # noqa: BLE001  (the port must be freed whatever happens)
        print(f"[live] finishing: {e!r}", flush=True)
    if hold and a.live_hold:
        print(f"[live] serving {server.url} {'until Ctrl-C' if a.live_hold < 0 else f'for {a.live_hold:g} s'}", flush=True)
        end = time.time() + (a.live_hold if a.live_hold > 0 else 1e12)
        try:
            while time.time() < end:
                cap.idle()                       # debug code, camera moves: on this (the simulation) thread
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
    cap.close()
    server.close()
    print("[live] closed", flush=True)


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
    ap.add_argument("--no-reads", action="store_true", help="do not take the plan's marker reads")
    ap.add_argument("--read-images", help="save every read's image here")
    ap.add_argument("--nominal", help="the layout the robot was taught in (fit of the reads; default: LAYOUT)")
    ap.add_argument("--located", help="write the layout corrected by the reads here")
    ap.add_argument("--out", required=True)
    lv = ap.add_argument_group("live view (rsf.live)")
    lv.add_argument("--live", metavar="[HOST:]PORT", help="serve a live view (default host 127.0.0.1)")
    lv.add_argument("--live-view", action="append", metavar="SPEC",
                    help="a view: 'overview' (default) or JSON {name, eye, target, fov_deg, width, height}; repeatable")
    lv.add_argument("--live-fps", type=float, default=5.0, help="rendered frames per wall-clock second (default 5)")
    lv.add_argument("--live-fps-interactive", type=float, default=20.0,
                    help="frames per second while a viewer moves a camera (default 20)")
    lv.add_argument("--live-size", type=int, nargs=2, metavar=("W", "H"), default=(960, 540))
    lv.add_argument("--live-pace", type=float, default=0.0,
                    help="no faster than this x real time (0: as fast as it runs)")
    lv.add_argument("--live-hold", type=float, default=0.0,
                    help="seconds to keep serving after the run (-1: until Ctrl-C)")
    lv.add_argument("--live-quality", type=int, default=80, help="JPEG quality")
    lv.add_argument("--live-record-dir", default="reports/live",
                    help="where recordings (MP4, from the viewer's Record) are saved (default reports/live)")
    lv.add_argument("--live-webrtc", action="store_true",
                    help="also send the views over WebRTC (VP8/H.264, UDP; needs aiortc); the viewer tries it first")
    lv.add_argument("--live-webrtc-ice", action="append", metavar="URL",
                    help="a STUN/TURN server for WebRTC: stun:host:3478, or JSON {urls, username, credential}; repeatable")
    lv.add_argument("--live-webrtc-codec", choices=("auto", "vp8", "h264"), default="auto",
                    help="WebRTC codec (auto: as negotiated with the browser, usually VP8)")
    lv.add_argument("--live-webrtc-kbps", type=float, default=0.0,
                    help="WebRTC top bitrate per viewer, kbit/s (0: aiortc's, VP8 1500, H.264 3000)")
    lv.add_argument("--live-webrtc-peers", type=int, default=8, help="at most this many WebRTC viewers (default 8)")
    lv.add_argument("--live-control", action="store_true", help="pause / step / pace / stop from the viewer (token)")
    lv.add_argument("--live-debug", action="store_true",
                    help="a Python console in the running simulation (token; implies --live-control)")
    lv.add_argument("--live-cert", help="serve HTTPS: certificate (PEM)")
    lv.add_argument("--live-key", help="serve HTTPS: private key (PEM)")
    ap.add_argument("--debugpy", metavar="[HOST:]PORT",
                    help="listen for a debugger (VS Code 'attach'; default host 127.0.0.1); needs debugpy")
    ap.add_argument("--debugpy-wait", action="store_true", help="wait for the debugger before simulating")
    a = ap.parse_args(argv)
    if (a.live_control or a.live_debug or a.live_cert or a.live_webrtc) and not a.live:
        ap.error("--live-control / --live-debug / --live-cert / --live-webrtc need --live")
    if a.debugpy:
        start_debugpy(a.debugpy, a.debugpy_wait)
    if a.live and a.backend == "mujoco" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")         # offscreen; MUJOCO_GL=osmesa without a GPU
    if a.backend == "isaac":
        # Isaac Sim's USD (pxr) loads with its app: start it before the layout, which may
        # read USD models (installing usd-core into Isaac's Python instead breaks Isaac)
        from rsf.sim.isaac_cell import start_app
        start_app(params(a.set, "isaac"))
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
    from rsf.sim import reads as RD
    setup = None if a.no_reads else RD.setup_for(world)
    if setup is not None and not RD.schedule(setup, traj["segments"]):
        setup = None                                  # the plan takes no reads
    if a.backend == "isaac":
        from rsf.sim.isaac_cell import IsaacCell
        cell = IsaacCell(world, occ, params(a.set, "isaac"), usd_path=a.usd, reads=setup)
    else:
        from rsf.sim.mujoco_cell import MujocoCell
        cell = MujocoCell(world, occ, params(a.set), reads=setup)
    reader = RD.Reader(setup, traj["segments"], cell, a.read_images) if setup is not None else None
    if reader is not None:
        print(f"reads: {len(reader.todo)} ({'detector ' + ('on' if setup.detect else 'missing: images only')})",
              flush=True)
    live = start_live(a, world, wf, cell, traj, t_end) if a.live else None
    if live:
        interrupt_on_hangup()
    hooks = [h for h in (reader, live[1] if live else None) if h is not None]
    on_sample = None if not hooks else (hooks[0] if len(hooks) == 1 else
                                        (lambda t, *x: [h(t, *x) for h in hooks] and None))
    t0 = time.time()
    try:
        rep = cell.run(traj, t_end=t_end, on_sample=on_sample,
                       progress=lambda t, w: print(f"  sim t={t:7.1f} s  wall {w:6.1f} s", flush=True))
    except BaseException as e:
        if live:
            from rsf.live.control import RunStopped
            if isinstance(e, RunStopped):
                print(f"STOPPED from the live view: {e}", flush=True)
                stop_live(a, *live, state="stopped", error=str(e))
                return 3
            if isinstance(e, KeyboardInterrupt):      # Ctrl-C, SIGTERM, SIGHUP: free the port, no hold
                stop_live(a, *live, state="interrupted", hold=False)
            else:
                stop_live(a, *live, state="error", error=repr(e))
        raise
    if a.report:                                      # the raw run, kept even if judging fails
        save_report(rep, a.report)
    res = J.judge(world, wf, rep, occ, traj=traj)
    res["meta"] = {"layout": a.layout, "workflow": a.workflow, "plan": a.plan, "sim_seconds": round(float(rep.t[-1]), 2),
                   "wall_seconds": round(time.time() - t0, 1), "backend": a.backend, "params": {k: v for k, v in vars(cell.p).items()}}
    if reader is not None:
        res["reads"] = {"summary": reader.summary(), "rows": reader.rows}
        sm = res["reads"]["summary"]
        print(f"READS {sm['found']}/{sm['reads']} found" + (
            f", board error max {sm['pos_err_mm']['max']} mm / {sm['rot_err_deg']['max']} deg" if sm.get("found") else ""),
            flush=True)
        if a.located and setup.detect is not None:
            nom = load_world(a.nominal) if a.nominal else world
            fits = RD.fit_entities(RD.setup_for(nom), reader.rows)
            out = RD.located_layout(a.nominal or a.layout, a.located, nom.name, fits, note="tools/sim_workflow.py")
            res["reads"]["located"] = {"layout": str(out), "fixtures": {
                k: {"reads": f["reads"], "estimated": f["estimated"]} for k, f in fits.items()}}
            for k, f in fits.items():
                print(f"  located {k}: {f['estimated']} ({f['reads']} reads)", flush=True)
            print(f"  -> {out}", flush=True)
    if live and live[1].console is not None and live[1].console.log:
        # debug code ran in this run: it may have changed it, so the verdict is not an acceptance result
        res["meta"]["live_debug"] = {"tainted": True, "log": live[1].console.log}
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
    if res["meta"].get("live_debug"):
        print(f"NOTE: debug code ran during this run ({len(res['meta']['live_debug']['log'])} times, meta.live_debug): "
              "not an acceptance result", flush=True)
    if live:
        stop_live(a, *live, state="done" if res["ok"] else "failed", ok=res["ok"], passed=n,
                  transfers=[{k: t[k] for k in ("src", "dst", "ok", "seat_mm", "slip_mm", "track_mm", "carry_tilt_deg")}
                             for t in res["transfers"]])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

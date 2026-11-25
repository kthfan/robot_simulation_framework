"""P7: the live view (rsf.live): hub, HTTP API, client, capture, and MuJoCo views
(Isaac's RTX views: tests/test_isaac.py)."""
from __future__ import annotations

import gzip
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rsf.live.capture import LiveCapture
from rsf.live.client import LiveClient
from rsf.live.hub import Hub, encode_jpeg
from rsf.live.server import LiveServer, parse_address
from rsf.sim.common import View, overview

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "cells" / "tm_tray"


def _image(v: int, w: int = 64, h: int = 48) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = v
    img[h // 4: h // 2, w // 4: w // 2] = 255
    return img


@pytest.fixture
def served():
    hub = Hub()
    srv = LiveServer(hub, "127.0.0.1", 0).start()
    yield hub, srv, LiveClient(srv.url, timeout=10)
    srv.close()


def test_parse_address():
    assert parse_address("8765") == ("127.0.0.1", 8765)
    assert parse_address(8765) == ("127.0.0.1", 8765)
    assert parse_address("0.0.0.0:9000") == ("0.0.0.0", 9000)


def test_api(served):
    hub, srv, c = served
    hub.set_info(backend="test")
    st = hub.stream("cam", width=64, height=48)
    st.publish(encode_jpeg(_image(10)), t=0.5)
    hub.update(state="running", t=0.5)
    hub.event("segment", name="free001")
    info = c.info()
    assert info["api"] == "rsf.live/1" and info["info"]["backend"] == "test"
    assert info["streams"]["cam"]["seq"] == 1 and "mjpeg" in info["transports"]
    assert c.status()["status"]["state"] == "running"
    assert [e["name"] for e in c.events_since(0)] == ["free001"]
    from PIL import Image
    assert Image.open(io.BytesIO(c.frame("cam"))).size == (64, 48)
    html = c._get("/").read().decode()
    assert "viewer.js" in html and c._get("/web/viewer.js").read()
    with pytest.raises(Exception, match="404"):
        c._get("/api/v1/nothing")

    @srv.route("GET", r"/api/v1/echo/(?P<x>\w+)")         # features add endpoints
    def echo(req, m):
        req.send_json({"x": m["x"], "q": req.query})
    assert c._json("/api/v1/echo/abc?k=1") == {"x": "abc", "q": {"k": "1"}}


def test_mjpeg_and_event_stream(served):
    hub, srv, c = served
    st = hub.stream("cam")
    st.publish(encode_jpeg(_image(10)), t=0.0)

    def later():
        time.sleep(0.3)
        st.publish(encode_jpeg(_image(200)), t=1.0)
        hub.event("done", ok=True)
        hub.update(state="done")
        time.sleep(0.3)
        hub.close()
    threading.Thread(target=later, daemon=True).start()
    frames = []
    for jpeg, head in c.mjpeg("cam"):
        frames.append((jpeg, head))
        if len(frames) == 2:
            break
    assert [h["x-frame-seq"] for _, h in frames] == ["1", "2"] and frames[1][1]["x-sim-time"] == "1.0"
    from PIL import Image
    assert np.asarray(Image.open(io.BytesIO(frames[1][0])))[..., 0].mean() > 150
    kinds = [k for k, _ in c.events()]                        # until the hub closes
    assert "done" in kinds and kinds.count("status") >= 1


class _FakeCell:
    def __init__(self):
        self.views, self.n = {}, 0

    def add_view(self, v):
        self.views[v.name] = v

    def render(self, name):
        self.n += 1
        return _image(self.n % 255, self.views[name].width, self.views[name].height)


def test_capture_frames_segments_and_status():
    hub = Hub()
    cell = _FakeCell()
    segs = [{"name": "a", "kind": "free", "t0": 0.0, "t1": 1.0}, {"name": "b", "kind": "line", "t0": 1.0, "t1": 2.0}]
    cap = LiveCapture(hub, cell, [View("v", (1, 0, 1), (0, 0, 0), width=32, height=24)], fps=1000.0,
                      segments=segs, t_end=2.0, status_hz=1000.0)
    cap.warmup()
    for t in np.arange(0.0, 2.0, 0.02):
        cap(float(t))
        time.sleep(0.002)
    cap.finish("done", ok=True)
    s = hub.snapshot()
    assert s["status"]["state"] == "done" and s["status"]["ok"] is True
    assert s["streams"]["v"]["seq"] >= 2 and hub.streams["v"].jpeg[:2] == b"\xff\xd8"
    ev = [e for e in hub.events if e["kind"] == "segment"]
    assert [e["name"] for e in ev] == ["a", "b"] and ev[1]["segment_kind"] == "line"
    assert [e["kind"] for e in hub.events][:2] == ["warmup", "run_start"]


def test_encoder_thread_stops():
    """The encoder is a Thread: its own fields must not shadow Thread's (Python 3.11 calls
    ``self._stop()`` in ``join``)."""
    from rsf.live.capture import _Encoder
    e = _Encoder(Hub(), 80)
    assert not isinstance(getattr(e, "_stop", None), bool)
    e.start()
    e.flush_and_stop()
    assert not e.is_alive()


def test_blank_frames_are_counted():
    hub = Hub()

    class Dark(_FakeCell):
        def render(self, name):
            return np.zeros((24, 32, 3), np.uint8)
    cap = LiveCapture(hub, Dark(), [View("v", (1, 0, 1), (0, 0, 0), width=32, height=24)], fps=1000.0)
    cap.warmup()
    cap(0.0)
    cap.finish()
    assert hub.status["blank_frames"] == 3              # warmup, the sample, the last frame


def test_capture_paces_to_real_time():
    hub = Hub()
    cap = LiveCapture(hub, _FakeCell(), [], pace=1.0)
    w = time.time()
    for t in np.arange(0.0, 0.5, 0.02):
        cap(float(t))
    assert time.time() - w >= 0.45
    cap.finish()


def test_view_spec(tmp_path):
    from rsf.model.layout import load_world
    w = load_world(CELL / "layouts" / "strict.yaml")
    v = overview(w)
    assert np.linalg.norm(np.subtract(v.eye, v.target)) > 2.0
    assert View.from_spec("overview", w).eye == v.eye
    u = View.from_spec({"name": "top", "eye": [0, 0, 3], "target": [0, 0.1, 0]})
    assert u.eye == (0.0, 0.0, 3.0) and u.width == 960
    with pytest.raises(ValueError):
        View.from_spec({"name": "x"})


# ── MuJoCo views ───────────────────────────────────────────────────────────
def _renders() -> bool:
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco/>")
        mujoco.Renderer(m, 8, 8).close()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _renders(), reason="no OpenGL context (set MUJOCO_GL)")
def test_mujoco_views_leave_the_run_unchanged():
    """Rendering every sample changes nothing in the run; the pictures show the arm move."""
    from rsf.model.layout import load_world
    from rsf.model.workflow import Workflow
    from rsf.sim.mujoco_cell import MujocoCell
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    traj = json.loads(gzip.decompress((CELL / "plans" / "strict_cycle1.json.gz").read_bytes()))
    cell = MujocoCell(world, occ)
    plain = cell.run(traj, t_end=2.0)
    hub = Hub()
    cap = LiveCapture(hub, cell, [overview(world, width=320, height=180)], fps=1000.0, segments=traj["segments"])
    shots = []
    cap.warmup()
    seen = cell.run(traj, t_end=2.0, on_sample=lambda t, *a: (cap(t, *a), shots.append(cell.render("overview").copy())))
    cap.finish()
    assert np.array_equal(plain.q, seen.q) and np.array_equal(plain.flange, seen.flange)
    for n in plain.objects:
        assert np.array_equal(plain.objects[n], seen.objects[n])
    assert shots[0].shape == (180, 320, 3) and shots[0].std() > 10
    assert np.abs(shots[-1].astype(int) - shots[0].astype(int)).mean() > 0.5        # the arm moved
    assert hub.streams["overview"].seq > 10


# ── control and the debug console ──────────────────────────────────────────
def _post(c, path, body, token=None):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(c.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with c._open(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_gated_routes_need_the_token():
    from rsf.live import control, debug
    hub = Hub()
    srv = LiveServer(hub, "127.0.0.1", 0, token="s3cret").start()
    try:
        c = LiveClient(srv.url)
        assert _post(c, "/api/v1/control", {"action": "pause"})[0] == 404        # not enabled on this run
        ctl, con = control.Controller(hub), debug.DebugConsole(hub)
        control.register(srv, ctl)
        debug.register(srv, con)
        assert set(c.info()["features"]) == {"control", "debug"}
        assert _post(c, "/api/v1/control", {"action": "pause"})[0] == 401
        assert _post(c, "/api/v1/control", {"action": "pause"}, "wrong")[0] == 401
        assert _post(c, "/api/v1/debug/exec", {"code": "1"}, "wrong")[0] == 401
        assert _post(c, "/api/v1/control", {"action": "pause"}, "s3cret") == (200, {"paused": True, "until": None,
                                                                                   "pace": None, "stop": False})
        assert _post(c, "/api/v1/control", {"action": "jump"}, "s3cret")[0] == 400
        threading.Thread(target=lambda: (time.sleep(0.2), con.pump(1.5)), daemon=True).start()
        code, j = _post(c, "/api/v1/debug/exec", {"code": "x = 6 * 7\nprint('hi')\nx + t", "wait": 5}, "s3cret")
        assert code == 200 and j["state"] == "done" and j["ok"] and j["value"] == "43.5" and j["output"] == "hi\n"
        assert [e["kind"] for e in hub.events][-1] == "debug" and con.log[0]["code"].startswith("x = 6")
    finally:
        srv.close()
    srv2 = LiveServer(Hub(), "127.0.0.1", 0).start()                       # no token: gated routes are off
    try:
        @srv2.route("POST", r"/api/v1/x", auth=True)
        def x(req, m):
            req.send_json({})
        assert _post(LiveClient(srv2.url), "/api/v1/x", {}, "anything")[0] == 403
    finally:
        srv2.close()


def test_debug_console_errors_and_exits_do_not_escape():
    from rsf.live.debug import DebugConsole
    con = DebugConsole(Hub(), {"k": 2})
    jobs = [con.submit(c) for c in ("k * 3", "1 / 0", "raise SystemExit(5)", "print('o', end='')")]
    assert con.pump(0.0) == 4
    r = [con.jobs[j["id"]] for j in jobs]
    assert r[0]["value"] == "6" and r[0]["ok"]
    assert not r[1]["ok"] and "ZeroDivisionError" in r[1]["error"]
    assert not r[2]["ok"] and "SystemExit" in r[2]["error"]
    assert r[3]["ok"] and r[3]["output"] == "o" and r[3]["value"] is None


def test_pause_step_resume_stop_hold_the_run():
    """A paused run waits between samples (debug code still runs), a step runs dt and
    pauses again, stop ends the run."""
    from rsf.live.control import Controller, RunStopped
    from rsf.live.debug import DebugConsole
    hub = Hub()
    ctl = Controller(hub)
    con = DebugConsole(hub)
    cap = LiveCapture(hub, _FakeCell(), [View("v", (1, 0, 1), (0, 0, 0), width=32, height=24)], fps=50.0,
                      controller=ctl, console=con)
    seen = []
    ctl.command("pause")

    def driver():
        time.sleep(0.2)
        assert seen == [0.0]                       # held at the first sample
        j = con.submit("t")
        assert con.wait(j["id"], 2.0)["value"] == "0.0"       # code runs while paused
        ctl.command("step", dt=0.1)
        time.sleep(0.2)
        assert seen[-1] == pytest.approx(0.1)       # stepped 0.1 s, then held
        ctl.command("resume")
        time.sleep(0.1)
        ctl.command("stop")
    th = threading.Thread(target=driver)
    th.start()
    with pytest.raises(RunStopped):
        for k in range(100000):
            t = round(k * 0.02, 6)
            seen.append(t)
            cap(t)
            time.sleep(0.001)
    th.join()
    assert seen[-1] > 0.1 and ctl.paused_s > 0.15
    assert [e["kind"] for e in hub.events if e["kind"] in ("paused", "control")][:3] == ["control", "paused", "control"]
    cap.finish("stopped")


def test_tls(tmp_path):
    """--live-cert/--live-key: the same API over HTTPS."""
    import shutil
    import ssl
    import subprocess
    import urllib.request
    if not shutil.which("openssl"):
        pytest.skip("no openssl")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    hub = Hub()
    srv = LiveServer(hub, "127.0.0.1", 0, ssl_context=ctx).start()
    errors = []
    srv.httpd.handle_error = (lambda orig: lambda *a: (errors.append(a), orig(*a)))(srv.httpd.handle_error)
    try:
        assert srv.url.startswith("https://")
        import socket
        stalled = socket.create_connection(srv.address)        # connects, never says hello
        cctx = ssl.create_default_context(cafile=str(cert))
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=cctx))
        w = time.time()
        with op.open(srv.url + "api/v1/info", timeout=10) as r:
            assert json.loads(r.read())["api"] == "rsf.live/1"
        assert time.time() - w < 5                                 # not held up by the stalled one
        hub.stream("cam").publish(encode_jpeg(_image(10)))
        raw = cctx.wrap_socket(socket.create_connection(srv.address), server_hostname="localhost")
        raw.sendall(b"GET /api/v1/video/cam/mjpeg HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assert b"multipart" in raw.recv(4096)
        raw.close()                                                # a viewer goes away mid-stream
        time.sleep(1.5)                                            # the server writes again, finds it gone
        stalled.close()
        assert errors == []
    finally:
        srv.close()


def test_unchanged_frames_are_counted():
    """A picture that never changes while the run goes on shows in the status."""
    class Still(_FakeCell):
        def render(self, name):
            return np.full((24, 32, 3), 100, np.uint8)
    hub = Hub()
    cap = LiveCapture(hub, Still(), [View("v", (1, 0, 1), (0, 0, 0), width=32, height=24)], fps=1000.0,
                      status_hz=1000.0)
    for t in np.arange(0.0, 0.2, 0.02):
        cap(float(t))
        time.sleep(0.002)
    assert hub.status["unchanged_frames"] == hub.status["frames"] - 1 >= 5
    cap.finish()


# ── cameras moved from the viewer (shared, no token) ───────────────────────
def test_view_angles_round_trip():
    from rsf.sim.common import view_angles, view_forward
    for eye, target in [((0, 0, 0), (1, 0, 0)), ((1, 2, 3), (0, 3, 2)), ((2.4, -2.1, 2.5), (0, 0.28, 1.06))]:
        y, p = view_angles(eye, target)
        d = np.subtract(target, eye)
        assert view_forward(y, p) == pytest.approx(d / np.linalg.norm(d))
    assert view_angles((0, 0, 0), (0, 1, 0)) == pytest.approx((90.0, 0.0))


class _MovableCell(_FakeCell):
    def __init__(self):
        super().__init__()
        self.moves = []

    def move_view(self, name, eye, target, fov_deg=None):
        self.moves.append((name, eye, target, fov_deg))
        from dataclasses import replace
        self.views[name] = replace(self.views[name], eye=eye, target=target, fov_deg=fov_deg)


def test_camera_api_moves_the_view_on_the_simulation_thread(served):
    from rsf.live import views
    hub, srv, c = served
    cell = _MovableCell()
    cap = LiveCapture(hub, cell, [View("v", (2, 0, 1), (0, 0, 1), width=32, height=24)], fps=1.0, fps_interactive=50.0)
    views.register(srv, cap)
    assert c.info()["features"]["camera"] == {"views": ["v"]}
    p = c._json("/api/v1/views")["v"]
    assert p["yaw"] == pytest.approx(180.0) and p["pitch"] == pytest.approx(0.0) and p["eye"] == [2, 0, 1]
    code, r = _post(c, "/api/v1/views/v/pose", {"eye": [1, 1, 2], "yaw": 90, "pitch": -30, "fov_deg": 60})   # no token
    assert code == 200 and r["target"] == pytest.approx([1, 1 + np.cos(np.radians(30)), 2 - 0.5])
    assert cell.moves == []                                   # not on the HTTP thread
    cap.warmup()
    n = cap.frames
    cap(0.0)
    assert cell.moves[-1][0] == "v" and cell.moves[-1][3] == 60.0 and cap.frames == n + 1
    assert hub.status["views"]["v"]["yaw"] == pytest.approx(90.0) and hub.status["views"]["v"]["pitch"] == pytest.approx(-30.0)
    # partial: only the yaw changes; the eye stays
    _post(c, "/api/v1/views/v/pose", {"yaw": 0})
    time.sleep(0.03)                                          # (frames at most fps_interactive)
    cap.idle()                                                # paused / after the run: a move draws at once
    assert cell.moves[-1][1] == (1.0, 1.0, 2.0) and cell.moves[-1][2] == pytest.approx((1 + np.cos(np.radians(30)), 1, 1.5))
    assert cap.frames == n + 2
    time.sleep(0.03)
    cap.idle()
    assert cap.frames == n + 2                                # nothing moved: no frame while still
    assert _post(c, "/api/v1/views/v/reset", {})[0] == 200
    cap.idle()
    assert cell.moves[-1][1:3] == ((2.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    for bad in ({"eye": [0, 0]}, {"eye": [0, 0, "x"]}, {"fov_deg": float("nan")}, {"eye": [1e6, 0, 0]}):
        assert _post(c, "/api/v1/views/v/pose", bad)[0] == 400, bad
    assert _post(c, "/api/v1/views/nope/pose", {"yaw": 1})[0] == 404
    p = c._json("/api/v1/views")["v"]
    assert _post(c, "/api/v1/views/v/pose", {"pitch": 120})[1]["target"][2] < p["eye"][2] + 1.0001   # clamped
    cap.finish()
    cap.close()


def test_frames_speed_up_while_a_camera_moves():
    cell = _MovableCell()
    cap = LiveCapture(Hub(), cell, [View("v", (2, 0, 1), (0, 0, 1), width=32, height=24)], fps=2.0,
                      fps_interactive=40.0)
    w = time.time()
    while time.time() - w < 0.5:
        cap(time.time() - w)
        time.sleep(0.005)
    still = cap.frames
    cap.request_view("v", yaw=10)
    w = time.time()
    while time.time() - w < 0.5:
        cap(time.time() - w)
        time.sleep(0.005)
    assert still <= 2 and cap.frames - still >= 10
    cap.close()


@pytest.mark.skipif(not _renders(), reason="no OpenGL context (set MUJOCO_GL)")
def test_mujoco_view_moves():
    from rsf.model.layout import load_world
    from rsf.sim.mujoco_cell import MujocoCell
    world = load_world(CELL / "layouts" / "strict.yaml")
    cell = MujocoCell(world)
    v = overview(world, width=160, height=90)
    cell.add_view(v)
    a = cell.render("overview").copy()
    cell.move_view("overview", (0.0, -3.0, 1.5), (0.0, 0.0, 1.0), 60.0)
    b = cell.render("overview")
    assert (np.abs(a.astype(int) - b.astype(int)).max(axis=2) > 10).mean() > 0.2
    cell.close_views()


# ── recording (MP4 on the server) ──────────────────────────────────────────
def _ffprobe(path):
    import shutil
    import subprocess
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,width,height,nb_read_frames,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def test_recording_follows_simulated_time_and_pauses_with_the_run(tmp_path, served):
    """Frames every 1/fps simulated seconds; a paused run adds none; saved as MP4 with a JSON."""
    from rsf.live import record
    from rsf.live.control import Controller
    if record.find_ffmpeg() is None:
        pytest.skip("no ffmpeg")
    hub, srv, c = served
    cell = _FakeCell()
    ctl = Controller(hub)
    rec = record.Recorder(hub, tmp_path, meta={"plan": "p"})
    cap = LiveCapture(hub, cell, [View("v", (1, 0, 1), (0, 0, 0), width=64, height=48)], fps=1.0,
                      controller=ctl, recorder=rec)
    record.register(srv, rec, cap)
    assert c.info()["features"]["record"]["available"]
    code, j = _post(c, "/api/v1/record/start", {"view": "v", "fps": 10})
    assert code == 200 and j["file"].startswith("v_") and j["frames"] == 0
    assert _post(c, "/api/v1/record/start", {"view": "v"})[0] == 400            # already recording
    assert _post(c, "/api/v1/record/start", {"view": "x"})[0] == 404
    for k in range(51):                                    # 1.0 s simulated at 50 samples/s -> 11 frames
        cap(round(k * 0.02, 6))
    assert rec.active["v"].frames == 11
    ctl.command("pause")                                   # the run pauses: the recording with it
    th = threading.Thread(target=lambda: cap(1.02), daemon=True)
    th.start()
    time.sleep(0.5)
    assert rec.active["v"].frames == 11
    ctl.command("resume")
    th.join(2)
    for k in range(52, 101):
        cap(round(k * 0.02, 6))
    assert rec.active["v"].frames == 21                    # 0 .. 2.0 s
    code, j = _post(c, "/api/v1/record/stop", {"view": "v"})
    assert code == 200 and j["saved"] and j["frames"] == 21 and j["t_start"] == 0.0 and j["t_end"] == pytest.approx(2.0)
    mp4 = tmp_path / j["file"]
    assert mp4.is_file() and not list(tmp_path.glob("*.part.mp4"))
    side = json.loads(mp4.with_suffix(".json").read_text())
    assert side["frames"] == 21 and side["plan"] == "p" and side["fps"] == 10.0
    info = _ffprobe(mp4)
    if info:
        assert (info["width"], info["height"], int(info["nb_read_frames"])) == (64, 48, 21)
        assert info["r_frame_rate"] == "10/1"
    with c._get("/api/v1/recordings/" + j["file"]) as r:
        assert r.read() == mp4.read_bytes()
    with pytest.raises(Exception, match="404"):
        c._get("/api/v1/recordings/..%2F..%2Fetc%2Fpasswd")
    # discard leaves nothing; a recording still open when the run ends is saved
    _post(c, "/api/v1/record/start", {"view": "v", "fps": 5})
    cap(2.02)
    assert _post(c, "/api/v1/record/discard", {"view": "v"})[1]["saved"] is None
    _post(c, "/api/v1/record/start", {"view": "v", "fps": 5})
    for k in range(102, 130):
        cap(round(k * 0.02, 6))
    cap.finish("done")
    assert len(list(tmp_path.glob("*.mp4"))) == 2 and [e["kind"] for e in hub.events].count("record_saved") == 2
    cap.close()


@pytest.mark.skipif(not _renders(), reason="no OpenGL context (set MUJOCO_GL)")
def test_record_a_mujoco_run(tmp_path):
    """The real thing: the first 3 s of the strict plan, recorded at 20 fps, run unchanged."""
    from rsf.live import record
    from rsf.model.layout import load_world
    from rsf.model.workflow import Workflow
    from rsf.sim.mujoco_cell import MujocoCell
    if record.find_ffmpeg() is None:
        pytest.skip("no ffmpeg")
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    traj = json.loads(gzip.decompress((CELL / "plans" / "strict_cycle1.json.gz").read_bytes()))
    cell = MujocoCell(world, occ)
    plain = cell.run(traj, t_end=3.0)
    hub = Hub()
    rec = record.Recorder(hub, tmp_path)
    cap = LiveCapture(hub, cell, [overview(world, width=320, height=180)], fps=2.0, recorder=rec)
    rec.start("overview", 320, 180, 20)
    seen = cell.run(traj, t_end=3.0, on_sample=cap)
    cap.finish()
    cap.close()
    assert np.array_equal(plain.q, seen.q)
    (f,) = tmp_path.glob("*.mp4")
    side = json.loads(f.with_suffix(".json").read_text())
    assert side["frames"] == 61 and side["error"] is None           # 0 .. 3.0 s at 20 fps
    info = _ffprobe(f)
    if info:
        assert info["codec_name"] in ("h264", "mpeg4") and int(info["nb_read_frames"]) == 61

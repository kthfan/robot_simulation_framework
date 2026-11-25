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

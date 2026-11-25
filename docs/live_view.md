# Live view (P7): watching a run from another computer

While `tools/sim_workflow.py` replays a plan, the simulator renders camera views
(Isaac Sim: RTX; MuJoCo: its OpenGL renderer) on the computer that runs it, and
serves them, with the run's status and events, over HTTP. A browser on any host
shows them; programs read the same API. Nothing is rendered in the browser.

![the viewer during the first strict transfer (MuJoCo)](images/live_view.png)

## Use

On the computer that simulates:

```bash
python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml \
    --plan cells/tm_tray/plans/strict_cycle1.json.gz --backend isaac --transfers 2 \
    --out reports/isaac/cycle1.json --live 8765 --live-hold 60
```

It prints `[live] http://localhost:8765/`. `--live 8765` listens on that
computer only (127.0.0.1). From another computer, either:

* **SSH tunnel** (no firewall change): run it through
  `tools/isaac/remote_isaac.sh run --view 8765 --cmd "python tools/sim_workflow.py ... --live 8765"`
  and open `http://localhost:8765/` on the main computer while it runs; or
* **the network**: `--live 0.0.0.0:8765`, open `http://<gpu-box>:8765/`.

| option | default | |
|---|---|---|
| `--live [HOST:]PORT` | off | serve the live view (host 127.0.0.1 unless given) |
| `--live-view SPEC` | `overview` | a view: `overview` (the whole cell) or JSON `{"name", "eye", "target", "fov_deg", "width", "height"}` (m, deg, px); repeatable |
| `--live-fps` | 5 | rendered frames per **wall-clock** second, all views together |
| `--live-size W H` | 960 540 | size of views without their own |
| `--live-pace X` | 0 | no faster than X × real time (0: as fast as it runs; MuJoCo runs faster than real time) |
| `--live-hold S` | 0 | keep serving S s after the run (-1: until Ctrl-C) |
| `--live-quality` | 80 | JPEG quality |

Example of a second view, from above the table:
`--live-view '{"name": "top", "eye": [0.0, 0.3, 3.0], "target": [0.0, 0.3, 0.8], "fov_deg": 40}'`.

MuJoCo renders offscreen with `MUJOCO_GL=egl` (set when there is no display);
`MUJOCO_GL=osmesa` without a GPU.

## Cost

Rendering runs on the simulation thread (Isaac's app loop and MuJoCo's GL context
need it), at most `--live-fps` times per wall-clock second, so its share of the run is
bounded; the status shows it (`render share`). JPEG encoding runs on another thread.

* MuJoCo (osmesa, no GPU, 960×540, 5 fps): 14 % of the wall time; at
  `--live-pace 1` the run kept real time (1.00×), 4.7 frames/s.
* Isaac Sim RTX on a Kaggle T4: one 1280×960 frame took 0.22–0.29 s
  (`render_probe.py`), so 5 fps would more than double the run there (physics alone
  ran at 0.75× real time). An RTX desktop GPU is several times faster; `--live-fps`
  and `--live-size` trade picture for speed. With a cold shader cache the first frame
  takes minutes: it is rendered before the run starts (`warmup` event), not in it.

Rendering does not change the run: MuJoCo gives bit-identical results with and
without it (`tests/test_live.py`). In Isaac the app update that renders does not
step physics (`/app/player/playSimulations` off during it, as Isaac's own
`SimulationContext.render`); `tests/test_isaac.py::test_live_view_renders_and_leaves_the_run`
checks that and that the pictures show the arm move. **That test has to run on a GPU**
(not yet run: no GPU here):

```bash
OMNI_KIT_ACCEPT_EULA=YES python -m pytest tests/test_isaac.py -k live_view -v
```

## API (`rsf.live`, `/api/v1`)

| endpoint | |
|---|---|
| `GET /` | the viewer (`src/rsf/live/web/`) |
| `GET /api/v1/info` | `api` (`rsf.live/1`), run info (backend, layout, plan, `t_end`), streams, `transports` |
| `GET /api/v1/status` | status: `state` (warmup, ready, running, done, failed, error), `t`, `rtf`, `segment`, `render_share`, `frames`, the result at the end |
| `GET /api/v1/events?since=N` | events after *N*: `warmup`, `run_start`, `segment`, `done` / `failed` / `error` |
| `GET /api/v1/stream` | Server-Sent Events: `status` snapshots (≤ `max_hz`, default 10) and `event` |
| `GET /api/v1/video/<view>/mjpeg` | MJPEG (`?fps=` caps it for one viewer); headers `X-Frame-Seq`, `X-Sim-Time` per frame |
| `GET /api/v1/video/<view>/frame.jpg` | the latest frame |

From Python (standard library only):

```python
from rsf.live.client import LiveClient
c = LiveClient("http://localhost:8765")
for kind, data in c.events():                 # until the run's server closes
    if kind == "segment":
        print(data["name"])
jpeg = c.frame("overview")
```

## Design: where features go

```
cell (Isaac / MuJoCo) ── add_view(View) / render(name) ──► LiveCapture (the on_sample hook)
                                                                │ RGB → JPEG (worker thread)
                                                                ▼
                                        Hub: video streams · status · event log
                                                                │
                              LiveServer: a route table (LiveServer.route) ── /api/v1/...
                                                                │
                      browser viewer (transports: mjpeg, jpeg)  ·  LiveClient / curl
```

* **A picture source** is a view on a cell; a backend offers `add_view` and `render`.
  More cameras, or one riding on the gripper, are more views.
* **A transport** reads the hub's latest frames: MJPEG now; WebRTC (aiortc/PyAV, H.264
  or VP8, signalling as `POST /api/v1/webrtc/offer`) and WebSocket fit beside it. The
  server lists its transports in `/api/v1/info`; the viewer has a matching table
  (`Viewer.transports` in `viewer.js`). WebRTC's media is UDP: it needs the two computers
  to reach each other directly (not through the SSH tunnel).
* **Endpoints** (control, recording, plots, a debug console) are routes added with
  `LiveServer.route`; the hub carries any status field and any event kind.

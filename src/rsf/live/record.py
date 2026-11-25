"""Recording a view to MP4 on the server (``rsf.live``): start, save, discard.

The video follows **simulated time**: a frame every ``1/fps`` simulated seconds, so it
plays the run at its real speed whatever the simulation's own speed, and it stops when
the run stops (paused from the viewer, or over): a paused run adds no frames. Frames
are the view as the viewers see it (a camera moved while recording moves in the video).

Encoding is ffmpeg's, fed raw RGB through a pipe from a writer thread: ``RSF_FFMPEG``,
else ``ffmpeg`` on the PATH, else the one of the ``imageio-ffmpeg`` package; H.264
(libx264) when it has it, else MPEG-4 part 2. The file is written as ``*.part.mp4`` and
renamed when saved, with a ``.json`` beside it (view, frames, simulated time span, run).
Recordings still open when the run ends are saved then.

``POST /api/v1/record/start`` ``{"view", "fps"}``, ``/stop`` ``{"view"}`` (save),
``/discard`` ``{"view"}``; ``GET /api/v1/record`` (state, saved files);
``GET /api/v1/recordings/<file>`` downloads one. No token: like the cameras, it does
not touch the simulation (it costs a render per frame).
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from rsf.live.hub import Hub

FILE = re.compile(r"[A-Za-z0-9_.-]+\.mp4")


def find_ffmpeg() -> Optional[str]:
    exe = os.environ.get("RSF_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def pick_codec(ffmpeg: str) -> list:
    try:
        enc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        enc = ""
    if " libx264 " in enc:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    return ["-c:v", "mpeg4", "-q:v", "3"]


class Recording:
    def __init__(self, ffmpeg: str, codec: list, path: Path, view: str, width: int, height: int, fps: float, meta: dict):
        self.view, self.fps, self.path, self.meta = view, float(fps), path, meta
        self.part = path.with_name(path.stem + ".part.mp4")
        self.width, self.height = int(width), int(height)
        self.frames = 0
        self.t_start = self.t_last = None
        self.next_t: Optional[float] = None
        self.wall_start = time.time()
        self.error: Optional[str] = None
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:g}", "-i", "-", "-an",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(self.part)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._q: queue.Queue = queue.Queue(maxsize=120)            # full: the simulation waits for ffmpeg
        self._thread = threading.Thread(target=self._write, name=f"rsf-rec-{view}", daemon=True)
        self._thread.start()

    def _write(self):
        while True:
            img = self._q.get()
            if img is None:
                return
            try:
                self.proc.stdin.write(img.tobytes())
            except (OSError, ValueError) as e:
                self.error = f"ffmpeg: {e!r}"
                return

    def add(self, img, t: float) -> None:
        import numpy as np
        if img.shape[0] != self.height or img.shape[1] != self.width:
            raise ValueError(f"frame {img.shape[1]}x{img.shape[0]}, recording {self.width}x{self.height}")
        self._q.put(np.ascontiguousarray(img[..., :3], dtype=np.uint8).copy())
        self.frames += 1
        if self.t_start is None:
            self.t_start = t
        self.t_last = t

    def close(self, save: bool) -> Optional[Path]:
        self._q.put(None)
        self._thread.join(60)
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.error = self.error or "ffmpeg did not finish"
        err = self.proc.stderr.read().decode(errors="replace").strip() if self.proc.stderr else ""
        if self.proc.returncode and not self.error:
            self.error = f"ffmpeg exit {self.proc.returncode}: {err[-500:]}"
        if not save or self.frames == 0 or self.error:
            self.part.unlink(missing_ok=True)
            return None
        self.part.replace(self.path)
        self.path.with_suffix(".json").write_text(json.dumps({**self.describe(), **self.meta}, indent=1))
        return self.path

    def describe(self) -> dict:
        return {"view": self.view, "file": self.path.name, "fps": self.fps, "frames": self.frames,
                "seconds": round(self.frames / self.fps, 2), "t_start": self.t_start, "t_end": self.t_last,
                "width": self.width, "height": self.height, "error": self.error}


class Recorder:
    def __init__(self, hub: Hub, out_dir, meta: Optional[dict] = None, ffmpeg: Optional[str] = None):
        self.hub = hub
        self.dir = Path(out_dir)
        self.meta = dict(meta or {})
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.codec = pick_codec(self.ffmpeg) if self.ffmpeg else []
        self.active: dict = {}                    # view -> Recording
        self.saved: list = []
        self._lock = threading.Lock()
        self.paused = False                       # the run is paused (shown; no frames come anyway)

    @property
    def available(self) -> bool:
        return self.ffmpeg is not None

    def start(self, view: str, width: int, height: int, fps: float = 20.0) -> dict:
        if not self.available:
            raise RuntimeError("no ffmpeg: install it, or set RSF_FFMPEG, or pip install imageio-ffmpeg")
        fps = float(fps)
        if not 1.0 <= fps <= 50.0:
            raise ValueError("fps: 1 to 50 (frames come at the simulation's samples, 50 a second)")
        with self._lock:
            if view in self.active:
                raise ValueError(f"{view} is already recording")
            self.dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = self.dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', view)}_{stamp}.mp4"
            n = 1
            while path.exists():
                n += 1
                path = path.with_name(f"{path.stem.rsplit('~', 1)[0]}~{n}.mp4")
            self.active[view] = Recording(self.ffmpeg, self.codec, path, view, width, height, fps, self.meta)
        self.hub.event("record_start", view=view, file=path.name, fps=fps)
        self._publish()
        return self.active[view].describe()

    def stop(self, view: str, save: bool = True) -> dict:
        with self._lock:
            rec = self.active.pop(view, None)
        if rec is None:
            raise KeyError(view)
        out = rec.close(save)
        d = rec.describe()
        d["saved"] = str(out) if out else None
        if out:
            self.saved.append(d)
        self.hub.event("record_saved" if out else "record_discarded", **{k: d[k] for k in ("view", "file", "frames",
                                                                                         "seconds", "error")})
        self._publish()
        return d

    def stop_all(self, save: bool = True) -> None:
        for v in list(self.active):
            self.stop(v, save)

    # ── on the simulation thread ───────────────────────────────────────────
    def due(self, t: float) -> list:
        """The views that need a frame at simulated time *t*."""
        out = []
        for v, r in list(self.active.items()):
            if r.next_t is None:
                r.next_t = t
            if t >= r.next_t - 1e-9:
                out.append(v)
        return out

    def feed(self, t: float, images: dict) -> None:
        for v in self.due(t):
            r = self.active.get(v)
            img = images.get(v)
            if r is None or img is None:
                continue
            while t >= r.next_t - 1e-9:            # one frame per 1/fps of simulated time
                r.add(img, t)
                r.next_t += 1.0 / r.fps
        self._publish(rate_limited=True)

    _last_pub = 0.0

    def _publish(self, rate_limited: bool = False) -> None:
        now = time.time()
        if rate_limited and now - self._last_pub < 0.25:
            return
        self._last_pub = now
        self.hub.update(recording={v: {**r.describe(), "wall_s": round(now - r.wall_start, 1)}
                                   for v, r in self.active.items()})

    def describe(self) -> dict:
        return {"available": self.available, "ffmpeg": self.ffmpeg,
                "codec": self.codec[1] if self.codec else None, "dir": str(self.dir.resolve()),
                "active": {v: r.describe() for v, r in self.active.items()}, "saved": self.saved}


def register(server, recorder: Recorder, capture) -> None:
    server.features["record"] = {"available": recorder.available, "codec": recorder.codec[1] if recorder.codec else None}

    def size(view):
        v = next((x for x in capture.views if x.name == view), None)
        if v is None:
            raise KeyError(view)
        return v.width, v.height

    @server.route("GET", r"/api/v1/record")
    def state(req, m):
        req.send_json(recorder.describe())

    @server.route("POST", r"/api/v1/record/(?P<act>start|stop|discard)")
    def act(req, m):
        b = req.json_body()
        view = str(b.get("view") or (capture.views[0].name if capture.views else ""))
        try:
            if m["act"] == "start":
                w, h = size(view)
                req.send_json(recorder.start(view, w, h, float(b.get("fps", 20.0))))
            else:
                req.send_json(recorder.stop(view, save=m["act"] == "stop"))
        except KeyError:
            req.send_json({"error": f"no view {view!r} (or it is not recording)"}, 404)
        except (ValueError, RuntimeError) as e:
            req.send_json({"error": str(e)}, 400)

    @server.route("GET", r"/api/v1/recordings/(?P<f>[^/]+)")
    def download(req, m):
        f = m["f"]
        p = recorder.dir / f
        if not FILE.fullmatch(f) or not p.is_file() or ".part." in f:
            return req.send_json({"error": "no such recording"}, 404)
        req.send_bytes(p.read_bytes(), "video/mp4", {"Content-Disposition": f'attachment; filename="{f}"'})

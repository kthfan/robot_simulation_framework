"""From a running cell to the hub: a callable for the cells' ``on_sample`` hook.

    cap = LiveCapture(hub, cell, [View.from_spec("overview", world)], fps=5, segments=traj["segments"])
    cap.warmup()                       # first frame before the run (RTX compiles shaders here)
    rep = cell.run(traj, on_sample=cap)
    cap.finish(ok=...)

Rendering happens on the simulation thread (the renderers need it: Isaac's app loop,
MuJoCo's GL context), at most *fps* times per wall-clock second, every view in one
go; JPEG encoding happens on a worker thread, so a frame costs the simulation only
its render. The cell must offer ``add_view(view)`` and ``render(name) -> (h, w, 3)
uint8 RGB or None``. Rendering only reads the simulation: the run is the same with
or without it (tests/test_live.py, tests/test_isaac.py).
"""
from __future__ import annotations

import bisect
import math
import threading
import time
from dataclasses import replace
from typing import Iterable, Optional

from rsf.live.hub import Hub, encode_jpeg


class _Encoder(threading.Thread):
    """The latest image of each stream -> JPEG -> the hub (older images are dropped)."""

    def __init__(self, hub: Hub, quality: int):
        super().__init__(name="rsf-live-jpeg", daemon=True)
        self.hub, self.quality = hub, quality
        self._slot: dict = {}
        self._cv = threading.Condition()
        self._halt = False        # (not _stop: threading.Thread has a _stop method)
        self.dropped = 0

    def submit(self, name: str, rgb, **meta):
        with self._cv:
            if name in self._slot:
                self.dropped += 1
            self._slot[name] = (rgb, meta)
            self._cv.notify()

    def run(self):
        while True:
            with self._cv:
                while not self._slot and not self._halt:
                    self._cv.wait()
                if not self._slot and self._halt:
                    return
                work, self._slot = self._slot, {}
            for name, (rgb, meta) in work.items():
                self.hub.streams[name].publish(encode_jpeg(rgb, self.quality), **meta)

    def flush_and_stop(self, timeout: float = 5.0):
        with self._cv:
            self._halt = True
            self._cv.notify()
        self.join(timeout)


class LiveCapture:
    def __init__(self, hub: Hub, cell, views: Iterable, fps: float = 5.0, pace: float = 0.0,
                 segments: Iterable = (), t_end: Optional[float] = None, quality: int = 80,
                 status_hz: float = 10.0, controller=None, console=None, fps_interactive: float = 20.0):
        self.hub, self.cell = hub, cell
        self.ctl, self.console = controller, console      # rsf.live.control / rsf.live.debug (optional)
        self.views = list(views)
        self.fps, self.pace = float(fps), float(pace)
        self.fps_interactive = max(float(fps_interactive), self.fps)   # while a viewer moves a camera
        self.home = {v.name: v for v in self.views}          # the views as given (reset)
        self._moves: dict = {}                               # view -> pending (eye, target, fov) from viewers
        self._moves_lock = threading.Lock()
        self._interact = -1e9                                # wall time of the last camera move
        self._dirty = False                                  # a camera moved: draw at once
        self.segments = sorted((s for s in segments if "t0" in s), key=lambda s: s["t0"])
        self._t0s = [s["t0"] for s in self.segments]
        self.t_end = t_end
        self.status_dt = 1.0 / status_hz
        self.enc = _Encoder(hub, quality)
        self.enc.start()
        self.render_s = 0.0                      # wall time spent rendering
        self.frames = 0
        self.blank = 0                           # frames with nothing in them (all dark): a renderer problem
        self.unchanged = 0                       # frames the same as the view's previous one (while running)
        self._last_sig: dict = {}
        self._seg = None
        self._wall0 = self._sim0 = None
        self._pace0 = None                       # (wall, sim) pacing counts from: restarts on resume / new pace
        self._pace_v = None
        self._last_frame = self._last_status = -1e9
        for v in self.views:
            cell.add_view(v)
            hub.stream(v.name, width=v.width, height=v.height, fov_deg=v.fov_deg, eye=v.eye, target=v.target)
        hub.update(views={v.name: self.view_pose(v.name) for v in self.views})

    # ── cameras (moved by viewers: rsf.live.views) ─────────────────────────
    def view_pose(self, name: str) -> dict:
        from rsf.sim.common import view_angles
        v = next(x for x in self.views if x.name == name)
        yaw, pitch = view_angles(v.eye, v.target)
        return {"name": name, "eye": [round(x, 5) for x in v.eye], "target": [round(x, 5) for x in v.target],
                "yaw": round(yaw, 4), "pitch": round(pitch, 4), "fov_deg": round(v.fov_deg, 3),
                "width": v.width, "height": v.height}

    def request_view(self, name: str, eye=None, target=None, yaw=None, pitch=None, fov_deg=None) -> dict:
        """Move a view's camera (any thread): applied on the simulation thread before the
        next frame. *eye* (m) with *target*, or with *yaw*/*pitch* (degrees); missing
        parts stay. Returns the pose asked for."""
        from rsf.sim.common import view_angles, view_forward
        if name not in self.home:
            raise KeyError(name)
        with self._moves_lock:
            cur = next(x for x in self.views if x.name == name)
            e0, t0, f0 = self._moves.get(name, (cur.eye, cur.target, cur.fov_deg))
            eye = e0 if eye is None else _vec3(eye, "eye")
            if target is not None:
                target = _vec3(target, "target")
            else:
                y0, p0 = view_angles(e0, t0)
                y = y0 if yaw is None else _num(yaw, "yaw")
                p = max(-89.9, min(89.9, p0 if pitch is None else _num(pitch, "pitch")))
                target = tuple(float(a + b) for a, b in zip(eye, view_forward(y, p)))
            fov = f0 if fov_deg is None else max(5.0, min(120.0, _num(fov_deg, "fov_deg")))
            if max(abs(a - b) for a, b in zip(eye, target)) < 1e-9:
                raise ValueError("eye and target coincide")
            self._moves[name] = (tuple(eye), tuple(target), float(fov))
            self._interact = time.time()
        return {"name": name, "eye": list(eye), "target": list(target), "fov_deg": fov}

    def reset_view(self, name: str) -> dict:
        h = self.home[name]
        return self.request_view(name, eye=h.eye, target=h.target, fov_deg=h.fov_deg)

    def _apply_moves(self) -> None:
        if not self._moves:
            return
        with self._moves_lock:
            moves, self._moves = self._moves, {}
        for name, (eye, target, fov) in moves.items():
            self.cell.move_view(name, eye, target, fov)
            i = next(k for k, x in enumerate(self.views) if x.name == name)
            self.views[i] = replace(self.views[i], eye=eye, target=target, fov_deg=fov)
            self.hub.stream(name, eye=eye, target=target, fov_deg=fov)
        self._dirty = True
        self.hub.update(views={v.name: self.view_pose(v.name) for v in self.views})

    def _frame_due(self, now: float) -> bool:
        if self._dirty:
            return now - self._last_frame >= 1.0 / self.fps_interactive
        rate = self.fps_interactive if now - self._interact < 1.0 else self.fps
        return now - self._last_frame >= 1.0 / rate

    # ── frames ─────────────────────────────────────────────────────────────
    def render_all(self, t: Optional[float]) -> None:
        w = time.time()
        for v in self.views:
            img = self.cell.render(v.name)
            if img is None or int(img[::8, ::8].max()) < 8:
                self.blank += 1
            if img is not None and t is not None:
                sig = img[::4, ::4].tobytes()
                if self._last_sig.get(v.name) == sig:
                    self.unchanged += 1
                self._last_sig[v.name] = sig
            if img is not None:
                self.enc.submit(v.name, img, t=None if t is None else round(float(t), 3))
        self._dirty = False
        self.render_s += time.time() - w
        self.frames += 1

    def warmup(self) -> None:
        """One frame of every view before the run (the scene as it starts)."""
        self.hub.update(state="warmup")
        w = time.time()
        self.render_all(None)
        self.hub.event("warmup", wall_s=round(time.time() - w, 2))
        self.hub.update(state="ready")

    # ── the hook ───────────────────────────────────────────────────────────
    def segment_at(self, t: float):
        i = bisect.bisect_right(self._t0s, t) - 1
        return self.segments[i] if 0 <= i < len(self.segments) and t <= self.segments[i].get("t1", t) else None

    def __call__(self, t, *args) -> None:
        now = time.time()
        if self._wall0 is None:
            self._wall0, self._sim0 = now, t
            self.hub.update(state="running", t_start_wall=now)
            self.hub.event("run_start", t=float(t))
        pace = self.ctl.pace if self.ctl is not None and self.ctl.pace is not None else self.pace
        if self._pace0 is None or pace != self._pace_v:
            self._pace0, self._pace_v = (now, t), pace
        if pace > 0:                              # no faster than pace x real time
            ahead = self._pace0[0] + (t - self._pace0[1]) / pace - now
            if ahead > 0:
                time.sleep(ahead)
                now = time.time()
        seg = self.segment_at(t)
        name = seg["name"] if seg else None
        if name != self._seg:
            self._seg = name
            if seg:
                self.hub.event("segment", name=name, segment_kind=seg.get("kind"), t0=seg["t0"], t1=seg.get("t1"))
        self._apply_moves()
        if self._frame_due(now):
            self._last_frame = now
            self.render_all(t)
        if now - self._last_status >= self.status_dt:
            self._last_status = now
            wall = now - self._wall0 - (self.ctl.paused_s if self.ctl is not None else 0.0)
            self.hub.update(t=round(float(t), 3), t_end=self.t_end, wall_s=round(now - self._wall0, 2),
                            rtf=round((t - self._sim0) / wall, 3) if wall > 0 else None,
                            segment=name, segment_kind=seg.get("kind") if seg else None,
                            render_share=round(self.render_s / wall, 3) if wall > 0 else None,
                            frames=self.frames, blank_frames=self.blank, unchanged_frames=self.unchanged,
                            dropped=self.enc.dropped)
        if self.console is not None:
            self.console.pump(t)
        if self.ctl is not None and self.ctl.gate(t, lambda: self.idle(t)):
            self._pace0 = (time.time(), t)        # pacing restarts after a pause

    def idle(self, t=None) -> None:
        """While the simulation waits (paused, or the run is over): run debug code, and
        draw when a camera moved (the scene itself stands still)."""
        if self.console is not None:
            self.console.pump(t)
        self._apply_moves()
        now = time.time()
        if self._dirty and self._frame_due(now):
            self._last_frame = now
            self.render_all(t)

    def finish(self, state: str = "done", **result) -> None:
        """The run is over: last frame, final status and a ``done`` (or other *state*) event.
        The views stay (cameras can still move, :meth:`idle`) until :meth:`close`."""
        try:
            self.render_all(None)
        except Exception:  # noqa: BLE001  (a closing app may not render)
            pass
        self.hub.update(state=state, blank_frames=self.blank, **result)
        self.hub.event(state, **result)

    def close(self) -> None:
        self.enc.flush_and_stop()
        close = getattr(self.cell, "close_views", None)
        if close:
            close()


def _num(x, what: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{what}: not a finite number")
    return v


def _vec3(x, what: str) -> tuple:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ValueError(f"{what}: three numbers")
    v = tuple(_num(a, what) for a in x)
    if max(abs(a) for a in v) > 1000.0:
        raise ValueError(f"{what}: beyond 1 km")
    return v

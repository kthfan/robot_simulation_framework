"""Run control from the live view: pause, resume, step, pace, stop (``--live-control``).

The commands come from HTTP threads and only set fields; the simulation thread acts on
them at its next sample (:meth:`Controller.gate`, called by the capture hook), so a
pause holds the run between two physics steps and changes nothing in it.

``POST /api/v1/control`` (token) with ``{"action": ...}``:

=========  ======================================================================
pause      hold at the next sample
resume     run on
step       run ``dt`` simulated seconds (default 0.1) from the current sample, then hold
pace       no faster than ``value`` x real time (0: as fast as it runs)
stop       end the run (the tool reports ``stopped``; no judging)
=========  ======================================================================
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from rsf.live.hub import Hub


class RunStopped(Exception):
    """The run was stopped from the live view."""


class Controller:
    def __init__(self, hub: Hub):
        self.hub = hub
        self.paused = False
        self.until: Optional[float] = None       # run to this simulated time, then pause
        self.stop = False
        self.pace: Optional[float] = None        # overrides the capture's pace when set
        self.t: Optional[float] = None           # the last sample's time
        self.paused_s = 0.0                      # wall time spent paused
        self.resumed = 0                         # counts resumptions (pacing restarts its clock)
        self._lock = threading.Lock()

    def command(self, action: str, **kw) -> dict:
        with self._lock:
            if action == "pause":
                self.paused, self.until = True, None
            elif action == "resume":
                self.paused, self.until = False, None
            elif action == "step":
                dt = float(kw.get("dt", 0.1))
                if dt <= 0:
                    raise ValueError("step: dt > 0")
                self.until = (self.t or 0.0) + dt
                self.paused = False
            elif action == "pace":
                v = float(kw.get("value", 0.0))
                if v < 0:
                    raise ValueError("pace: value >= 0")
                self.pace = v
            elif action == "stop":
                self.stop, self.paused = True, False
            else:
                raise ValueError(f"unknown action {action!r} (pause, resume, step, pace, stop)")
        self.hub.event("control", action=action, **{k: v for k, v in kw.items() if k in ("dt", "value")})
        self.hub.update(paused=self.paused, step_until=self.until, pace=self.pace, stopping=self.stop)
        return {"paused": self.paused, "until": self.until, "pace": self.pace, "stop": self.stop}

    def gate(self, t: float, idle: Callable[[], None]) -> bool:
        """At every sample (simulation thread): stop, or hold while paused, running *idle*
        meanwhile (debug code, frames, status). True when it held."""
        self.t = t
        if self.stop:
            raise RunStopped(f"stopped at t={t:.3f} s")
        if self.until is not None and t >= self.until - 1e-9:
            with self._lock:
                self.paused, self.until = True, None
            self.hub.update(paused=True, step_until=None)
        if not self.paused:
            return False
        self.hub.event("paused", t=round(float(t), 3))
        self.hub.update(paused=True, t=round(float(t), 3))
        w = time.time()
        while self.paused and not self.stop:
            idle()
            time.sleep(0.02)
        self.paused_s += time.time() - w
        self.resumed += 1
        if self.stop:
            raise RunStopped(f"stopped at t={t:.3f} s")
        self.hub.update(paused=False)
        return True


def register(server, ctl: Controller) -> None:
    server.features["control"] = {"actions": ["pause", "resume", "step", "pace", "stop"]}

    @server.route("POST", r"/api/v1/control", auth=True)
    def control(req, m):
        body = req.json_body()
        try:
            req.send_json(ctl.command(str(body.pop("action", "")), **body))
        except ValueError as e:
            req.send_json({"error": str(e)}, 400)

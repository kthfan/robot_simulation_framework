"""A Python console in the running simulation (``--live-debug``): code sent from the
viewer or any client runs **in the simulation process, on the simulation thread**,
between two physics steps (or while paused, :mod:`rsf.live.control`), with the run's
objects at hand (``cell``, ``world``, ``traj``, ``hub``, ``np``, ``t``; the tool adds
more). Isaac Sim and MuJoCo calls are only safe there, not on an HTTP thread.

It is full access to the computer as the user that runs the simulation: off unless
asked for, every call needs the token, and every piece of code is logged (events
``debug``, and the run's report). Code that changes the simulation makes the run's
judging meaningless; the report says that debug code ran. Code that never returns
holds the simulation (nothing can interrupt it from outside).

``POST /api/v1/debug/exec`` (token) ``{"code": "...", "wait": 10}``: the job, done
or still queued; ``GET /api/v1/debug/job/<id>``; ``GET /api/v1/debug/log``. The last
expression's value comes back as its ``repr`` (like a REPL), with stdout and stderr.
"""
from __future__ import annotations

import ast
import collections
import contextlib
import io
import threading
import time
import traceback
from typing import Optional

from rsf.live.hub import Hub

MAX_TEXT = 100_000


def _clip(s: str) -> str:
    return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + f"\n... ({len(s) - MAX_TEXT} more characters)"


class DebugConsole:
    def __init__(self, hub: Hub, namespace: Optional[dict] = None):
        self.hub = hub
        self.ns = {"__name__": "__rsf_debug__", **(namespace or {})}
        self.jobs: dict = {}
        self.log: list = []                       # every job: id, wall, t, code, ok
        self._queue = collections.deque()
        self._cv = threading.Condition()
        self._n = 0

    def submit(self, code: str, origin: str = "") -> dict:
        with self._cv:
            self._n += 1
            job = {"id": self._n, "state": "queued", "code": code, "origin": origin, "submitted": time.time()}
            self.jobs[job["id"]] = job
            self._queue.append(job)
            return dict(job)

    def wait(self, job_id: int, timeout: float = 10.0) -> dict:
        end = time.time() + timeout
        with self._cv:
            job = self.jobs[job_id]
            while job["state"] != "done":
                left = end - time.time()
                if left <= 0:
                    break
                self._cv.wait(left)
            return dict(job)

    def pump(self, t: Optional[float] = None) -> int:
        """Run the queued code (simulation thread). Returns how many ran."""
        n = 0
        while self._queue:
            with self._cv:
                if not self._queue:
                    break
                job = self._queue.popleft()
                job["state"] = "running"
            self._run(job, t)
            n += 1
        return n

    def _run(self, job: dict, t: Optional[float]) -> None:
        self.ns["t"] = t
        out = io.StringIO()
        res = {"ok": False, "value": None, "error": None}
        w = time.time()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                tree = ast.parse(job["code"], "<debug>", "exec")
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                exec(compile(tree, "<debug>", "exec"), self.ns)          # noqa: S102  (the feature)
                value = eval(compile(last, "<debug>", "eval"), self.ns) if last else None  # noqa: S307
                if value is not None:
                    res["value"] = _clip(repr(value))
                    self.ns["_"] = value
            res["ok"] = True
        except BaseException:  # noqa: BLE001  (SystemExit / KeyboardInterrupt from code must not end the run)
            res["error"] = _clip(traceback.format_exc())
        res["output"] = _clip(out.getvalue())
        res["seconds"] = round(time.time() - w, 4)
        entry = {"id": job["id"], "wall": w, "t": None if t is None else round(float(t), 3), "code": job["code"],
                 "origin": job["origin"], "ok": res["ok"]}
        with self._cv:
            job.update(res, state="done", t=entry["t"])
            self.log.append(entry)
            self._cv.notify_all()
        self.hub.event("debug", id=job["id"], ok=res["ok"], t=entry["t"], code=job["code"][:500])


def register(server, console: DebugConsole) -> None:
    server.features["debug"] = {"exec": "/api/v1/debug/exec"}

    @server.route("POST", r"/api/v1/debug/exec", auth=True)
    def exec_(req, m):
        body = req.json_body()
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            return req.send_json({"error": "code: a non-empty string"}, 400)
        job = console.submit(code, origin=req.client_address[0])
        req.send_json(console.wait(job["id"], float(body.get("wait", 10.0))))

    @server.route("GET", r"/api/v1/debug/job/(?P<id>\d+)", auth=True)
    def job(req, m):
        j = console.jobs.get(int(m["id"]))
        req.send_json(dict(j) if j else {"error": "no such job"}, 200 if j else 404)

    @server.route("GET", r"/api/v1/debug/log", auth=True)
    def log(req, m):
        req.send_json(console.log)

"""The live API from Python (standard library only).

    from rsf.live.client import LiveClient
    c = LiveClient("http://localhost:8765")
    print(c.info()["streams"])
    for kind, data in c.events():           # "status" and every event, as they happen
        ...
    for jpeg, headers in c.mjpeg("overview"):
        ...
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterator


class LiveClient:
    def __init__(self, url: str = "http://localhost:8765", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        # a viewer talks to its LAN or a tunnel, never through an HTTP proxy
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def _get(self, path: str):
        return self._open(self.url + path, timeout=self.timeout)

    def _json(self, path: str):
        with self._get(path) as r:
            return json.loads(r.read())

    def info(self) -> dict:
        return self._json("/api/v1/info")

    def status(self) -> dict:
        return self._json("/api/v1/status")

    def events_since(self, seq: int = 0) -> list:
        return self._json(f"/api/v1/events?since={int(seq)}")

    def frame(self, stream: str) -> bytes:
        """The latest JPEG of *stream*."""
        with self._get(f"/api/v1/video/{stream}/frame.jpg") as r:
            return r.read()

    def events(self, since: int = 0, max_hz: float = 10.0) -> Iterator[tuple]:
        """``(kind, data)`` from the event stream: ``status`` snapshots and the events;
        ends when the run's server closes."""
        with self._get(f"/api/v1/stream?since={int(since)}&max_hz={max_hz}") as r:
            kind, data = "message", []
            for raw in r:
                line = raw.decode().rstrip("\r\n")
                if not line:
                    if data:
                        d = json.loads("\n".join(data))
                        yield (d.get("kind", kind) if kind == "event" else kind), d
                    kind, data = "message", []
                elif line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())

    def mjpeg(self, stream: str, fps: float = 0.0, repeats: bool = False) -> Iterator[tuple]:
        """``(jpeg, headers)`` of every frame of *stream* as it is published (the server
        repeats the latest frame for browsers; *repeats* keeps those)."""
        q = f"?fps={fps}" if fps else ""
        last = None
        with self._get(f"/api/v1/video/{stream}/mjpeg{q}") as r:
            while True:
                line = r.readline()
                if not line:
                    return
                if not line.startswith(b"--rsfframe"):
                    continue
                headers = {}
                while True:
                    h = r.readline().decode().strip()
                    if not h:
                        break
                    k, _, v = h.partition(":")
                    headers[k.strip().lower()] = v.strip()
                jpeg = r.read(int(headers["content-length"]))
                if repeats or headers.get("x-frame-seq") != last:
                    last = headers.get("x-frame-seq")
                    yield jpeg, headers

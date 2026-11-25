"""What a live run publishes: video streams (latest JPEG of each), the run status
(one dict, merged updates) and an event log. Publishers never block on viewers:
a viewer that is slow just sees the latest frame."""
from __future__ import annotations

import threading
import time
from typing import Optional


class VideoStream:
    """The latest frame of one picture source (a rendered view)."""

    def __init__(self, name: str, hub: "Hub", **meta):
        self.name = name
        self.meta = dict(meta)
        self._hub = hub
        self.seq = 0
        self.jpeg: Optional[bytes] = None
        self.frame_meta: dict = {}
        self._times: list = []                   # wall times of recent frames (frame rate)

    def publish(self, jpeg: bytes, **meta) -> int:
        now = time.time()
        with self._hub.cond:
            self.seq += 1
            self.jpeg = jpeg
            self.frame_meta = {**meta, "seq": self.seq, "wall": now}
            self._times = [t for t in self._times if now - t < 2.0] + [now]
            self._hub.cond.notify_all()
            return self.seq

    @property
    def fps(self) -> float:
        ts = self._times
        return 0.0 if len(ts) < 2 else (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)

    def describe(self) -> dict:
        return {**self.meta, "name": self.name, "seq": self.seq, "fps": round(self.fps, 2),
                **{k: v for k, v in self.frame_meta.items() if k in ("t", "wall")}}

    def wait(self, after: int, timeout: float = 10.0):
        """``(seq, jpeg, frame_meta)`` of the first frame newer than *after* (``None``
        on timeout or when the hub closes)."""
        end = time.time() + timeout
        with self._hub.cond:
            while self.seq <= after and not self._hub.closed:
                left = end - time.time()
                if left <= 0:
                    return None
                self._hub.cond.wait(left)
            if self.seq <= after:
                return None
            return self.seq, self.jpeg, dict(self.frame_meta)


class Hub:
    def __init__(self, max_events: int = 10000):
        self.cond = threading.Condition()
        self.info: dict = {}                     # static: backend, layout, plan, ...
        self.status: dict = {"state": "starting"}
        self.status_seq = 0
        self.events: list = []                   # {"seq", "kind", "wall", ...}
        self.event_seq = 0
        self.streams: dict = {}
        self.closed = False
        self._max = max_events

    def set_info(self, **kw) -> None:
        with self.cond:
            self.info.update(kw)

    def update(self, **kw) -> None:
        """Merge *kw* into the status."""
        with self.cond:
            self.status.update(kw)
            self.status_seq += 1
            self.cond.notify_all()

    def event(self, kind: str, /, **data) -> dict:
        with self.cond:
            self.event_seq += 1
            e = {**data, "seq": self.event_seq, "kind": kind, "wall": time.time()}
            self.events.append(e)
            del self.events[:-self._max]
            self.cond.notify_all()
            return e

    def stream(self, name: str, **meta) -> VideoStream:
        with self.cond:
            if name not in self.streams:
                self.streams[name] = VideoStream(name, self, **meta)
            else:
                self.streams[name].meta.update(meta)
            return self.streams[name]

    def snapshot(self) -> dict:
        with self.cond:
            return {"status": dict(self.status), "status_seq": self.status_seq, "event_seq": self.event_seq,
                    "streams": {n: s.describe() for n, s in self.streams.items()}}

    def events_since(self, seq: int) -> list:
        with self.cond:
            return [e for e in self.events if e["seq"] > seq]

    def wait(self, status_seq: int, event_seq: int, timeout: float = 10.0) -> bool:
        """Until the status or the events are newer than given (False on timeout)."""
        end = time.time() + timeout
        with self.cond:
            while self.status_seq <= status_seq and self.event_seq <= event_seq and not self.closed:
                left = end - time.time()
                if left <= 0:
                    return False
                self.cond.wait(left)
            return True

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


def encode_jpeg(rgb, quality: int = 80) -> bytes:
    """An ``(h, w, 3)`` uint8 RGB image as JPEG: OpenCV if it is there (Isaac Sim brings
    it), else Pillow."""
    import numpy as np
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass
    import io

    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=int(quality))
    return b.getvalue()

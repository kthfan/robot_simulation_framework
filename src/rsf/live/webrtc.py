"""WebRTC video for the live view (``rsf.live``): every view as VP8 or H.264 over RTP,
encoded from the rendered RGB (not from the JPEG). Compared with MJPEG: a fraction of the
bandwidth, the picture shown as soon as it arrives (no frame held back by the browser),
and the browser's jitter buffer instead of TCP's retransmissions. Optional: it needs
``aiortc`` (``pip install aiortc``; it brings PyAV and its codecs) in the simulator's
Python; the rest of ``rsf.live`` stays standard library only.

======================================  ==========================================================
``POST /api/v1/webrtc/offer``           ``{"sdp", "type": "offer", "stream": view}`` -> ``{"sdp",
                                        "type": "answer", "id"}`` (the browser's offer, recvonly)
``POST /api/v1/webrtc/<id>/close``      ends that peer (a viewer switching away)
``GET /api/v1/webrtc``                  the peers: stream, state, codec, frames sent
======================================  ==========================================================

Signalling is one request, without trickle ICE: the browser gathers its candidates,
sends its offer; the answer carries every candidate of this computer (aiortc gathers
them in ``setLocalDescription``). No token, like the MJPEG stream: it does not touch the
simulation.

**Network.** The media is UDP between the two computers, on ports aioice picks (no
fixed range): the signalling may go through an SSH tunnel, the media cannot. The browser
must reach one of this computer's addresses (all interfaces but loopback are offered),
or a STUN/TURN server must be given (``ice_servers``, to both ends). The viewer falls
back to MJPEG when the connection does not come up.

**Frames.** A peer's track waits for the stream's next frame (the capture's rate) and
repeats the latest one after ``idle_s`` without one (a paused run: the browser's
decoder gets a keyframe after a loss, a new viewer gets a picture). Odd sizes lose
their last row/column (YUV 4:2:0). Each peer has its own encoder (aiortc encodes per
sender) on aiortc's worker threads; ``max_peers`` bounds them, and a peer that has not
connected ``connect_s`` after its offer is closed (a page gone before its answer).
"""
from __future__ import annotations

import asyncio
import fractions
import json
import threading
import time
import uuid
from typing import Optional

from rsf.live.hub import Hub

CLOCK = 90000                                 # RTP video clock (Hz)


def missing() -> Optional[str]:
    """Why WebRTC is not available here (``None``: it is)."""
    try:
        import aiortc  # noqa: F401
        import av  # noqa: F401
    except ImportError as e:
        return f"{e.name or 'aiortc'} is not installed (pip install aiortc)"
    return None


def parse_ice(spec) -> dict:
    """An ICE server: ``"stun:host:3478"``, ``"turn:host:3478"`` or JSON
    ``{"urls": ..., "username": ..., "credential": ...}`` (the browser's format)."""
    if isinstance(spec, dict):
        d = dict(spec)
    else:
        s = str(spec).strip()
        d = json.loads(s) if s.startswith("{") else {"urls": s}
    urls = d.get("urls")
    if not urls or not all(str(u).split(":", 1)[0] in ("stun", "stuns", "turn", "turns")
                           for u in (urls if isinstance(urls, list) else [urls])):
        raise ValueError(f"ICE server {spec!r}: urls must be stun:, stuns:, turn: or turns:")
    return {k: d[k] for k in ("urls", "username", "credential") if k in d}


def _track_class():
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError

    class HubTrack(MediaStreamTrack):
        """The frames of one hub stream as a video track (its latest RGB image)."""
        kind = "video"

        def __init__(self, stream, idle_s: float = 1.0):
            super().__init__()
            self.stream, self.idle_s = stream, float(idle_s)
            self.seq = 0
            self.frames = 0
            self._t0 = None
            self._pts = -1
            self._last = 0.0
            self._loop = asyncio.get_running_loop()
            self._wake = asyncio.Event()
            self._cb = lambda seq: self._loop.call_soon_threadsafe(self._wake.set)
            stream.listen(self._cb)

        async def recv(self):
            import av
            import numpy as np
            s = self.stream
            while True:
                if self.readyState != "live" or s._hub.closed:
                    raise MediaStreamError
                self._wake.clear()
                if s.rgb is not None and (s.seq > self.seq or time.monotonic() - self._last >= self.idle_s):
                    break
                wait = self.idle_s - (time.monotonic() - self._last) if s.rgb is not None else self.idle_s
                try:
                    await asyncio.wait_for(self._wake.wait(), max(wait, 0.01))
                except asyncio.TimeoutError:
                    pass
            self.seq, rgb = s.seq, s.rgb
            h, w = rgb.shape[0] & ~1, rgb.shape[1] & ~1
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[:h, :w, :3], dtype=np.uint8), format="rgb24")
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now
            self._pts = max(self._pts + 1, int((now - self._t0) * CLOCK))
            frame.pts, frame.time_base = self._pts, fractions.Fraction(1, CLOCK)
            self._last = now
            self.frames += 1
            return frame

        def stop(self):
            self.stream.unlisten(self._cb)
            super().stop()

    return HubTrack


class WebRTC:
    """The peers of one live server, on an asyncio loop of their own (a thread)."""

    def __init__(self, hub: Hub, ice_servers=(), codec: str = "auto", max_peers: int = 8,
                 idle_s: float = 1.0, max_kbps: Optional[float] = None, connect_s: float = 30.0):
        why = missing()
        if why:
            raise RuntimeError(f"WebRTC: {why}")
        self.hub = hub
        self.ice_servers = [parse_ice(s) for s in ice_servers]
        self.codec = (codec or "auto").lower()
        if self.codec not in ("auto", "vp8", "h264"):
            raise ValueError("codec: auto, vp8 or h264")
        self.max_peers, self.idle_s, self.connect_s = int(max_peers), float(idle_s), float(connect_s)
        self.max_kbps = max_kbps
        if max_kbps:
            _set_bitrate(float(max_kbps) * 1000.0)
        self.peers: dict = {}                     # id -> {"pc", "track", "stream", "since", "state"}
        self._Track = None
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="rsf-live-webrtc", daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _call(self, coro, timeout: float = 30.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    # ── from the HTTP threads ─────────────────────────────────────────────
    def offer(self, sdp: str, kind: str, stream: str) -> dict:
        if kind != "offer" or not isinstance(sdp, str) or not sdp.startswith("v="):
            raise ValueError("an SDP offer: {\"sdp\": \"v=0...\", \"type\": \"offer\", \"stream\": view}")
        s = self.hub.streams.get(stream)
        if s is None:
            raise KeyError(stream)
        if len(self.peers) >= self.max_peers:
            raise OverflowError(f"{self.max_peers} WebRTC viewers already (max_peers): use MJPEG")
        return self._call(self._offer(sdp, s))

    def close_peer(self, pid: str) -> bool:
        if pid not in self.peers:
            return False
        self._call(self._close(pid), timeout=10.0)
        return True

    def describe(self) -> dict:
        out = {}
        for pid, p in list(self.peers.items()):
            out[pid] = {"stream": p["stream"], "state": p["pc"].connectionState, "codec": p.get("codec"),
                        "frames": p["track"].frames, "seconds": round(time.time() - p["since"], 1)}
        return {"peers": out, "max_peers": self.max_peers, "codec": self.codec, "ice_servers": len(self.ice_servers)}

    def close(self) -> None:
        if not self.loop.is_running():
            return
        try:
            self._call(self._close_all(), timeout=10.0)
        except Exception:  # noqa: BLE001  (closing: free the port whatever happens)
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5.0)

    # ── on the loop ───────────────────────────────────────────────────────
    async def _offer(self, sdp: str, stream) -> dict:
        from aiortc import (RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpSender,
                            RTCSessionDescription)
        if self._Track is None:
            self._Track = _track_class()
        conf = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in self.ice_servers])
        pc = RTCPeerConnection(conf)
        pid = uuid.uuid4().hex[:12]
        track = self._Track(stream, self.idle_s)
        self.peers[pid] = {"pc": pc, "track": track, "stream": stream.name, "since": time.time()}

        @pc.on("connectionstatechange")
        async def _state():
            if pc.connectionState in ("connected", "failed", "closed"):
                self.hub.event("webrtc", peer=pid, stream=stream.name, state=pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self._close(pid)

        def _reap():                              # never connected (the page went away before its answer)
            if pid in self.peers and pc.connectionState != "connected":
                self.hub.event("webrtc", peer=pid, stream=stream.name, state="timeout")
                asyncio.ensure_future(self._close(pid))
        self.loop.call_later(self.connect_s, _reap)
        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            video = [t for t in pc.getTransceivers() if t.kind == "video"]
            if not video:
                raise ValueError("the offer has no video (addTransceiver('video', {direction: 'recvonly'}))")
            sender = pc.addTrack(track)
            if self.codec != "auto":
                want = f"video/{self.codec}"
                caps = [c for c in RTCRtpSender.getCapabilities("video").codecs
                        if c.mimeType.lower() in (want, "video/rtx")]
                tr = next(t for t in pc.getTransceivers() if t.sender is sender)
                tr.setCodecPreferences(caps)
            await pc.setLocalDescription(await pc.createAnswer())
        except Exception:
            await self._close(pid)
            raise
        ans = pc.localDescription
        codec = _answer_codec(ans.sdp)
        self.peers[pid]["codec"] = codec
        self.hub.event("webrtc", peer=pid, stream=stream.name, state="offered", codec=codec)
        return {"sdp": ans.sdp, "type": ans.type, "id": pid, "codec": codec}

    async def _close(self, pid: str) -> None:
        p = self.peers.pop(pid, None)
        if p is None:
            return
        p["track"].stop()
        await p["pc"].close()

    async def _close_all(self) -> None:
        for pid in list(self.peers):
            await self._close(pid)


def _answer_codec(sdp: str) -> Optional[str]:
    """The first video codec of an answer (the one the sender uses)."""
    pts, maps = None, {}
    for line in sdp.splitlines():
        if line.startswith("m=video"):
            pts = line.split()[3:]
        elif line.startswith("a=rtpmap:"):
            pt, _, enc = line[9:].partition(" ")
            maps[pt] = enc.split("/")[0]
    for pt in pts or []:
        if maps.get(pt, "").lower() not in ("rtx", "red", "ulpfec", ""):
            return maps[pt]
    return None


def _set_bitrate(bps: float) -> None:
    """aiortc's encoders start at a fixed bitrate and follow the receiver's estimate
    (REMB) within a module-wide range: set the top (and the start) of that range."""
    from aiortc.codecs import h264, vpx
    for m in (vpx, h264):
        m.MAX_BITRATE = int(max(bps, m.MIN_BITRATE))
        m.DEFAULT_BITRATE = int(min(m.MAX_BITRATE, max(m.DEFAULT_BITRATE, bps / 2)))


def register(server, hub: Hub, prefer: bool = True, **kw) -> WebRTC:
    """WebRTC on *server*: the routes, ``webrtc`` in its transports (first when *prefer*:
    the viewer tries it first and falls back to MJPEG), ``features["webrtc"]`` (the ICE
    servers the browser should use). *kw*: :class:`WebRTC`'s."""
    rtc = WebRTC(hub, **kw)
    if "webrtc" not in server.transports:
        server.transports.insert(0, "webrtc") if prefer else server.transports.append("webrtc")
    server.features["webrtc"] = {"ice_servers": rtc.ice_servers, "codec": rtc.codec, "max_peers": rtc.max_peers}
    server.on_close.append(rtc.close)

    @server.route("POST", r"/api/v1/webrtc/offer")
    def offer(req, m):
        try:
            b = req.json_body()
            req.send_json(rtc.offer(b.get("sdp"), b.get("type"), str(b.get("stream", ""))))
        except KeyError as e:
            req.send_json({"error": f"no stream {e.args[0]!r}"}, 404)
        except OverflowError as e:
            req.send_json({"error": str(e)}, 503)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            req.send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001  (an SDP aiortc cannot answer, a timeout)
            req.send_json({"error": f"WebRTC: {e!r}"}, 500)

    @server.route("POST", r"/api/v1/webrtc/(?P<pid>[0-9a-f]{1,32})/close")
    def close(req, m):
        req.send_json({"closed": rtc.close_peer(m["pid"])})

    @server.route("GET", r"/api/v1/webrtc")
    def peers(req, m):
        req.send_json(rtc.describe())

    return rtc


async def receive(url: str, stream: str, frames: int = 1, timeout: float = 20.0, ice_servers=()):
    """Program side (and tests): connect to *url*'s WebRTC like the viewer does and return
    *frames* decoded frames of *stream* as ``(h, w, 3)`` RGB arrays, and the answer."""
    import urllib.request

    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**parse_ice(s)) for s in ice_servers]))
    got: asyncio.Future = asyncio.get_running_loop().create_future()

    @pc.on("track")
    def _track(t):
        if not got.done():
            got.set_result(t)
    try:
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        body = json.dumps({"sdp": pc.localDescription.sdp, "type": "offer", "stream": stream}).encode()
        open_ = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

        def post():
            r = urllib.request.Request(url.rstrip("/") + "/api/v1/webrtc/offer", data=body,
                                       headers={"Content-Type": "application/json"})
            with open_(r, timeout=timeout) as resp:
                return json.loads(resp.read())
        ans = await asyncio.get_running_loop().run_in_executor(None, post)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        track = await asyncio.wait_for(got, timeout)
        out = []
        end = time.monotonic() + timeout
        while len(out) < frames:
            f = await asyncio.wait_for(track.recv(), max(end - time.monotonic(), 0.1))
            out.append(f.to_ndarray(format="rgb24"))
        return out, ans
    finally:
        await pc.close()

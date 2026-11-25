"""The live API over HTTP (standard library only) and the viewer page.

========================================  ====================================================
``GET /``                                 the viewer (``web/index.html``); ``/web/<file>``
``GET /api/v1/info``                      api version, run info, streams, transports
``GET /api/v1/status``                    status snapshot (state, sim time, segment, ...)
``GET /api/v1/events?since=N``            the event log after event *N*
``GET /api/v1/stream``                    Server-Sent Events: ``status`` snapshots, ``event`` (``kind`` inside)
``GET /api/v1/video/<name>/mjpeg``        the stream as MJPEG (``?fps=`` caps it per viewer; the
                                          latest frame is repeated each second while idle)
``GET /api/v1/video/<name>/frame.jpg``    the latest frame
========================================  ====================================================

Further endpoints are added with :meth:`LiveServer.route` (a regex on the path, a
handler taking the request handler and the match), so features need no change here;
``auth=True`` routes need the server's token (``Authorization: Bearer <token>``), and
answer 403 when the server has none (the feature is off). ``ssl_context`` serves HTTPS.
"""
from __future__ import annotations

import errno
import hmac
import json
import mimetypes
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rsf.live import API
from rsf.live.hub import Hub

WEB = Path(__file__).resolve().parent / "web"


def parse_address(spec, default_host: str = "127.0.0.1") -> tuple:
    """``"8765"``, ``"0.0.0.0:8765"`` or ``8765`` -> ``(host, port)``."""
    s = str(spec)
    host, _, port = s.rpartition(":")
    return (host or default_host), int(port)


class _HTTPServer(ThreadingHTTPServer):
    """Threads per connection; TLS handshakes in the connection's thread (a client that
    never finishes one must not hold up the others); viewers that go away are not errors."""
    daemon_threads = True
    ssl_context = None

    def finish_request(self, request, client_address):
        if self.ssl_context is None:
            return super().finish_request(request, client_address)
        request.settimeout(15.0)
        try:
            tls = self.ssl_context.wrap_socket(request, server_side=True)
        except OSError:                           # not TLS, or gone during the handshake
            return None
        tls.settimeout(None)
        try:
            super().finish_request(tls, client_address)
        finally:
            try:
                tls.close()
            except OSError:
                pass

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], OSError):  # a closed tab, a reset, a TLS EOF
            return
        super().handle_error(request, client_address)


class LiveServer:
    def __init__(self, hub: Hub, host: str = "127.0.0.1", port: int = 8765, token: str = None,
                 ssl_context=None):
        self.hub = hub
        self.routes: list = []                    # (method, compiled regex, handler, auth)
        self.transports = ["mjpeg", "jpeg"]
        self.features: dict = {}                  # name -> description, listed in /api/v1/info
        self.token = token
        self._add_defaults()
        server = self

        class Handler(_Handler):
            live = server
        try:
            self.httpd = _HTTPServer((host, port), Handler)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                raise OSError(e.errno, f"port {port} is in use on {host}: another program, or an earlier run still "
                                       f"serving (find it: ss -ltnp | grep :{port}; on the GPU box: tools/isaac/remote_isaac.sh stop)") from None
            raise
        self.httpd.ssl_context = ssl_context
        self.tls = ssl_context is not None
        self._thread = None

    @property
    def address(self) -> tuple:
        return self.httpd.server_address[:2]

    @property
    def url(self) -> str:
        h, p = self.address
        return f"{'https' if self.tls else 'http'}://{'localhost' if h in ('127.0.0.1', '0.0.0.0', '::') else h}:{p}/"

    def route(self, method: str, pattern: str, auth: bool = False):
        """Decorator: ``handler(req, match)`` for *method* on paths matching *pattern*
        (anchored). ``req`` is the :class:`http.server.BaseHTTPRequestHandler` with
        ``send_json``, ``send_bytes``, ``query``, ``body`` and ``json_body()`` helpers.
        *auth*: only with the token."""
        def deco(fn):
            self.routes.append((method.upper(), re.compile(pattern + r"\Z"), fn, auth))
            return fn
        return deco

    def authorized(self, req) -> bool:
        if not self.token:
            return False
        h = req.headers.get("Authorization", "")
        return h.startswith("Bearer ") and hmac.compare_digest(h[7:].strip().encode(), self.token.encode())

    def start(self) -> "LiveServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="rsf-live", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.hub.close()                           # ends the streaming responses
        self.httpd.shutdown()
        self.httpd.server_close()

    # ── the built-in endpoints ────────────────────────────────────────────
    def _add_defaults(self):
        hub = self.hub

        @self.route("GET", r"/")
        def index(req, m):
            req.send_file(WEB / "index.html")

        @self.route("GET", r"/web/(?P<f>[A-Za-z0-9_.-]+)")
        def web(req, m):
            req.send_file(WEB / m["f"])

        @self.route("GET", r"/api/v1/info")
        def info(req, m):
            snap = hub.snapshot()
            req.send_json({"api": API, "info": hub.info, "streams": snap["streams"], "transports": self.transports,
                           "features": self.features, "status": snap["status"]})

        @self.route("GET", r"/api/v1/status")
        def status(req, m):
            req.send_json(hub.snapshot())

        @self.route("GET", r"/api/v1/events")
        def events(req, m):
            req.send_json(hub.events_since(int(req.query.get("since", 0))))

        @self.route("GET", r"/api/v1/stream")
        def sse(req, m):
            max_hz = float(req.query.get("max_hz", 10))
            req.send_response(200)
            req.send_header("Content-Type", "text/event-stream")
            req.send_header("Cache-Control", "no-cache")
            req.send_header("Connection", "close")
            req.end_headers()
            st, ev = -1, int(req.query.get("since", 0))
            while True:
                snap = hub.snapshot()
                out = []
                for e in hub.events_since(ev):
                    out.append(f"event: event\nid: {e['seq']}\ndata: {json.dumps(e, default=float)}\n\n")
                    ev = e["seq"]
                if snap["status_seq"] != st:
                    st = snap["status_seq"]
                    out.append(f"event: status\ndata: {json.dumps(snap, default=float)}\n\n")
                req.wfile.write(("".join(out) or ": keep-alive\n\n").encode())
                req.wfile.flush()
                if hub.closed:
                    return
                time.sleep(1.0 / max_hz)               # at most max_hz messages a second
                hub.wait(st, ev, timeout=15.0)

        @self.route("GET", r"/api/v1/video/(?P<name>[^/]+)/frame\.jpg")
        def frame(req, m):
            s = hub.streams.get(m["name"])
            if s is None or s.jpeg is None:
                return req.send_json({"error": f"no frame of {m['name']!r} yet"}, 404)
            req.send_bytes(s.jpeg, "image/jpeg", {"X-Frame-Seq": s.seq})

        @self.route("GET", r"/api/v1/video/(?P<name>[^/]+)/mjpeg")
        def mjpeg(req, m):
            s = hub.streams.get(m["name"])
            if s is None:
                return req.send_json({"error": f"no stream {m['name']!r}"}, 404)
            min_dt = 1.0 / float(req.query["fps"]) if "fps" in req.query else 0.0
            req.send_response(200)
            req.send_header("Content-Type", "multipart/x-mixed-replace; boundary=rsfframe")
            req.send_header("Cache-Control", "no-cache")
            req.send_header("Connection", "close")
            req.end_headers()
            # Browsers show a part only once the next part starts (Chrome: measured), so a
            # new frame goes out again 60 ms later unless a newer one came: no frame waits
            # for the next (a still scene, a camera moved while paused). The latest frame
            # is also repeated every second while idle, and twice on connect.
            req.wfile.write(b"--rsfframe\r\n")

            def part(seq, jpeg, meta):
                head = (f"Content-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n"
                        f"X-Frame-Seq: {seq}\r\nX-Sim-Time: {meta.get('t', '')}\r\n\r\n")
                req.wfile.write(head.encode() + jpeg + b"\r\n--rsfframe\r\n")
                req.wfile.flush()
            seq, last, fresh = 0, 0.0, False
            if s.jpeg is not None:
                seq, jpeg, meta = s.seq, s.jpeg, dict(s.frame_meta)
                part(seq, jpeg, meta)
                part(seq, jpeg, meta)
            while True:
                got = s.wait(seq, timeout=0.06 if fresh else 1.0)
                if got is None:
                    if hub.closed:
                        return
                    if s.jpeg is not None:
                        part(s.seq, s.jpeg, dict(s.frame_meta))
                    fresh = False
                    continue
                seq, jpeg, meta = got
                part(seq, jpeg, meta)
                fresh = True
                if min_dt:
                    time.sleep(max(0.0, min_dt - (time.time() - last)))
                    last = time.time()


class _Handler(BaseHTTPRequestHandler):
    live: LiveServer = None
    protocol_version = "HTTP/1.1"
    server_version = "rsf-live"

    def log_message(self, fmt, *args):          # quiet
        pass

    def _dispatch(self, method):
        u = urlparse(self.path)
        self.query = {k: v[-1] for k, v in parse_qs(u.query).items()}
        n = int(self.headers.get("Content-Length") or 0)
        self.body = self.rfile.read(n) if n else b""
        for meth, rx, fn, auth in self.live.routes:
            m = rx.match(u.path)
            if m and meth == method:
                if auth and not self.live.authorized(self):
                    return self.send_json({"error": "this needs the token (Authorization: Bearer ...)"
                                           if self.live.token else "not enabled on this run"},
                                          401 if self.live.token else 403)
                try:
                    return fn(self, m)
                except OSError:                    # the client went away (also TLS EOF): not an error here
                    return None
        self.send_json({"error": f"no route {method} {u.path}"}, 404)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def send_bytes(self, data: bytes, ctype: str, headers=None, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)

    def json_body(self) -> dict:
        return json.loads(self.body or b"{}")

    def send_json(self, obj, code: int = 200):
        self.send_bytes(json.dumps(obj, default=float).encode(), "application/json", code=code)

    def send_file(self, path: Path):
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), ctype)

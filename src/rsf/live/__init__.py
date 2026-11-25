"""Live view of a running simulation (P7): pictures the simulator renders (Isaac Sim
RTX, or MuJoCo), its status and its events, served over HTTP to a browser on any
host and to other programs.

* :mod:`rsf.live.hub`: what is published — the latest frame of every video stream,
  the run status, an event log. Thread-safe; the simulation only publishes.
* :mod:`rsf.live.server`: the HTTP API (``/api/v1/...``) and the viewer page, on the
  standard library only (no packages to install, air-gapped hosts included).
  Routes are a table (:meth:`LiveServer.route`), so features add endpoints.
* :mod:`rsf.live.capture`: the bridge from a cell's ``on_sample`` hook: renders the
  views at a frame rate (wall clock), encodes JPEG off the simulation thread,
  updates the status, optionally paces the run to real time.
* :mod:`rsf.live.client`: the API from Python (``LiveClient``).

Video goes out as MJPEG (``multipart/x-mixed-replace``): any browser shows it, it
crosses an SSH tunnel and needs nothing beyond HTTP. Transports are named in
``/api/v1/info`` (``transports``) so others (WebRTC, WebSocket) can be added
without changing the hub or the capture. See ``docs/live_view.md``.
"""
API = "rsf.live/1"

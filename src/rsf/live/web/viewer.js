// rsf live viewer: frames from /api/v1/video/<name>/<transport> (or WebRTC), status and
// events from /api/v1/stream (SSE). Transports are a table: add one with
// Viewer.transports.name = {attach(img, stream), detach(img)} and list it in the
// server's /api/v1/info "transports" (the first one both know is used; the choice is kept).
"use strict";
const Viewer = {
  transports: {
    webrtc: {                     // VP8/H.264 over UDP (rsf.live.webrtc); falls back to MJPEG when it does not connect
      _gen: 0,
      attach(img, stream) {
        const gen = ++this._gen, video = $("vid");
        img.hidden = true;
        video.hidden = false;
        $("nosignal").style.display = "flex";
        const ice = (Viewer.features.webrtc || {}).ice_servers || [];
        const pc = this._pc = new RTCPeerConnection({ iceServers: ice });
        this._id = null;
        const fail = (why) => { if (gen === this._gen) Viewer.fallback(`WebRTC: ${why}`); };
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.ontrack = (e) => { video.srcObject = new MediaStream([e.track]); video.play().catch(() => {}); };
        pc.onconnectionstatechange = () => { if (pc.connectionState === "failed") fail("the connection failed"); };
        video.onplaying = () => { clearTimeout(this._timer); $("nosignal").style.display = "none"; };
        this._timer = setTimeout(() => fail("no picture within 8 s (UDP blocked between the two computers? " +
                                            "an SSH tunnel carries only HTTP)"), 8000);
        (async () => {
          await pc.setLocalDescription(await pc.createOffer());
          await iceGathered(pc, 3000, !ice.length);
          const r = await fetch("/api/v1/webrtc/offer", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sdp: pc.localDescription.sdp, type: "offer", stream }) });
          const j = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
          if (gen !== this._gen) { closePeer(j.id); return; }
          this._id = j.id;
          await pc.setRemoteDescription({ type: j.type, sdp: j.sdp });
        })().catch((e) => fail(e.message));
        let prev = null;
        this._stats = setInterval(async () => {
          if (gen !== this._gen) return;
          prev = await rtcStats(pc, prev);
          if (prev && gen === this._gen) setTransportNote(prev.text);
        }, 1000);
      },
      detach(img) {
        this._gen++;
        clearTimeout(this._timer);
        clearInterval(this._stats);
        if (this._pc) this._pc.close();
        if (this._id) closePeer(this._id);
        this._pc = this._id = null;
        const video = $("vid");
        video.onplaying = null;
        video.srcObject = null;
        video.hidden = true;
        img.hidden = false;
      },
    },
    mjpeg: {
      attach(img, stream) { img.src = `/api/v1/video/${encodeURIComponent(stream)}/mjpeg?t=${Date.now()}`; },
      detach(img) { img.removeAttribute("src"); },
    },
    jpeg: {                       // polling, for proxies that buffer multipart responses
      attach(img, stream) {
        const tick = () => { img.src = `/api/v1/video/${encodeURIComponent(stream)}/frame.jpg?t=${Date.now()}`; };
        img.onload = () => { this._t = setTimeout(tick, 200); };
        tick();
      },
      detach(img) { clearTimeout(this._t); img.onload = null; img.removeAttribute("src"); },
    },
  },
  stream: null, transport: null, events: 0, features: {}, available: [],
};

// The offer goes out with the candidates (no trickle). Without ICE servers only host candidates
// come, at once, but browsers may report "complete" only seconds later: go 200 ms after the last one.
function iceGathered(pc, ms, hostOnly) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((done) => {
    let quiet = null;
    const end = () => { clearTimeout(cap); clearTimeout(quiet); done(); };
    const cap = setTimeout(end, ms);
    pc.addEventListener("icegatheringstatechange", () => { if (pc.iceGatheringState === "complete") end(); });
    pc.addEventListener("icecandidate", (e) => {
      if (!e.candidate) return end();
      if (hostOnly) { clearTimeout(quiet); quiet = setTimeout(end, 200); }
    });
  });
}

function closePeer(id) {
  if (id) fetch(`/api/v1/webrtc/${id}/close`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: "{}", keepalive: true }).catch(() => {});
}

async function rtcStats(pc, prev) {
  let rep;
  try { rep = await pc.getStats(); } catch (e) { return prev; }
  let inb = null, pair = null;
  const codecs = {};
  rep.forEach((r) => {
    if (r.type === "inbound-rtp" && r.kind === "video") inb = r;
    if (r.type === "codec") codecs[r.id] = r.mimeType;
    if (r.type === "candidate-pair" && r.nominated && r.state === "succeeded") pair = r;
  });
  if (!inb) return prev;
  const now = performance.now(), bytes = inb.bytesReceived || 0;
  const kbps = prev && now > prev.t ? 8 * (bytes - prev.bytes) / (now - prev.t) : null;
  const parts = [(codecs[inb.codecId] || "").replace("video/", "") || "webrtc"];
  if (inb.frameWidth) parts.push(`${inb.frameWidth}×${inb.frameHeight}`);
  if (inb.framesPerSecond !== undefined) parts.push(`${fmt(inb.framesPerSecond, 0)} fps`);
  if (kbps !== null) parts.push(`${fmt(kbps, 0)} kb/s`);
  if (pair && pair.currentRoundTripTime !== undefined) parts.push(`rtt ${fmt(1000 * pair.currentRoundTripTime, 0)} ms`);
  if (inb.packetsLost) parts.push(`lost ${inb.packetsLost}`);
  return { t: now, bytes, text: parts.join(" · ") };
}

function setTransportNote(text, bad = false) {
  const el = $("tstats");
  el.textContent = text;
  el.classList.toggle("fail", bad);
}

function storedTransport() {
  try { return localStorage.getItem("rsf-live-transport"); } catch (e) { return null; }
}

function useTransport(name, why) {
  const img = $("img");
  if (Viewer.stream && Viewer.transport) Viewer.transports[Viewer.transport].detach(img);
  Viewer.transport = name;
  $("tsel").value = name;
  setTransportNote(why || "", !!why);
  if (Viewer.stream) Viewer.transports[name].attach(img, Viewer.stream);
}

// WebRTC did not come up: the next transport (MJPEG), for this page only (the choice stays)
Viewer.fallback = (why) => {
  const next = Viewer.available.find((t) => t !== "webrtc");
  if (!next || Viewer.transport !== "webrtc") return;
  useTransport(next, `${why}; showing ${next}`);
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined) ? "–" : (typeof v === "number" ? v.toFixed(d) : String(v));

async function init() {
  const params = new URLSearchParams(location.search);
  const info = await (await fetch("/api/v1/info")).json();
  const i = info.info || {};
  Viewer.info = i;
  $("run").textContent = [i.backend, i.layout, i.plan].filter(Boolean).join(" · ");
  const names = Object.keys(info.streams);
  Viewer.features = info.features || {};
  Viewer.available = info.transports.filter((t) => Viewer.transports[t] && (t !== "webrtc" || window.RTCPeerConnection));
  if (!Viewer.available.length) Viewer.available = ["jpeg"];
  const ok = (t) => t && Viewer.available.includes(t);
  const want = params.get("transport"), kept = storedTransport();
  Viewer.transport = ok(want) ? want : ok(kept) ? kept : Viewer.available[0];
  const tsel = $("tsel");
  for (const t of Viewer.available) tsel.add(new Option(t, t));
  tsel.value = Viewer.transport;
  tsel.onchange = () => {
    try { localStorage.setItem("rsf-live-transport", tsel.value); } catch (e) { /* storage blocked */ }
    useTransport(tsel.value);
  };
  const img = $("img");
  img.addEventListener("load", () => { $("nosignal").style.display = "none"; });
  img.addEventListener("error", () => { $("nosignal").style.display = "flex"; });
  const nav = $("streams");
  for (const n of names) {
    const b = document.createElement("button");
    b.textContent = n;
    b.onclick = () => show(n);
    nav.appendChild(b);
  }
  setupGated(info.features || {});
  if ((info.features || {}).camera) Camera.init($("frame"));
  else $("camera").hidden = true;
  if ((info.features || {}).record) setupRecord(info.features.record);
  if (names.length) show(params.get("stream") && names.includes(params.get("stream")) ? params.get("stream") : names[0]);
  listen();
}

// ── recording (MP4 on the server; follows simulated time, so a paused run records nothing) ──
function setupRecord(feat) {
  $("record").hidden = false;
  if (!feat.available) {
    $("recstart").disabled = true;
    $("recstate").textContent = "recording off: no ffmpeg on the server";
    return;
  }
  const act = async (what, body) => {
    const r = await fetch(`/api/v1/record/${what}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ view: Viewer.stream, ...body }) });
    const j = await r.json();
    if (!r.ok) $("recstate").textContent = j.error || `HTTP ${r.status}`;
    if (what !== "start") refreshRecordings();
    return j;
  };
  $("recstart").onclick = () => act("start", { fps: Number($("recfps").value) });
  $("recsave").onclick = () => act("stop", {});
  $("recdiscard").onclick = () => { if (confirm("Discard this recording?")) act("discard", {}); };
  refreshRecordings();
}

async function refreshRecordings() {
  const r = await (await fetch("/api/v1/record")).json();
  const box = $("reclist");
  box.replaceChildren();
  if (!r.saved.length) { box.textContent = `saved to ${r.dir} on the server (${r.codec})`; return; }
  box.append(`saved in ${r.dir}: `);
  r.saved.slice().reverse().forEach((s, i) => {
    if (i) box.append(" · ");
    const a = document.createElement("a");
    a.href = `/api/v1/recordings/${encodeURIComponent(s.file)}`;
    a.textContent = s.file;
    a.title = `${s.frames} frames, ${s.seconds} s, simulated ${fmt(s.t_start, 1)}–${fmt(s.t_end, 1)} s`;
    box.append(a, ` (${s.seconds} s)`);
  });
}

function renderRecording(s) {
  if ($("record").hidden) return;
  const rec = (s.recording || {})[Viewer.stream];
  const on = !!rec;
  $("recstart").hidden = on;
  $("recsave").hidden = !on;
  $("recdiscard").hidden = !on;
  $("recfps").disabled = on;
  const st = $("recstate");
  st.classList.toggle("on", on);
  if (on) {
    const why = s.paused ? " · paused with the run" : (["done", "failed", "stopped", "interrupted"].includes(s.state) ? " · the run is over" : "");
    st.textContent = `● REC ${fmt(rec.seconds, 1)} s video · ${rec.frames} frames · simulated ${fmt(rec.t_start, 1)}–${fmt(rec.t_end, 1)} s${why}`;
  } else if (st.classList.contains("on") || st.textContent.startsWith("●")) {
    st.textContent = "";
  }
}

// ── control and console (only when the run enabled them; both need the token) ──
function token() { return $("token").value.trim(); }

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json",
    "Authorization": `Bearer ${token()}` }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function setupGated(features) {
  if (!features.control && !features.debug) return;
  $("auth").hidden = false;
  const m = location.hash.match(/token=([^&]+)/);
  let saved = null;
  try { saved = sessionStorage.getItem("rsf-live-token"); } catch (e) { /* storage blocked */ }
  $("token").value = m ? decodeURIComponent(m[1]) : (saved || "");
  if (m) history.replaceState(null, "", location.pathname + location.search);    // keep it out of the address bar
  $("token").addEventListener("change", () => { try { sessionStorage.setItem("rsf-live-token", token()); } catch (e) { /* */ } });
  try { if (token()) sessionStorage.setItem("rsf-live-token", token()); } catch (e) { /* */ }
  if (features.control) {
    $("controls").hidden = false;
    const pace = String(Viewer.info.pace ?? 0), sel = $("pace");
    if (![...sel.options].some((o) => o.value === pace)) sel.add(new Option(`${pace}×`, pace));
    sel.value = pace;
    for (const b of $("controls").querySelectorAll("button")) {
      b.onclick = async () => {
        const body = { action: b.dataset.act };
        if (b.dataset.dt) body.dt = Number(b.dataset.dt);
        if (body.action === "stop" && !confirm("Stop the run? It ends without judging.")) return;
        try { await post("/api/v1/control", body); $("ctlmsg").textContent = ""; }
        catch (e) { $("ctlmsg").textContent = e.message; }
      };
    }
    $("pace").onchange = () => post("/api/v1/control", { action: "pace", value: Number($("pace").value) })
      .catch((e) => { $("ctlmsg").textContent = e.message; });
  }
  if (features.debug) {
    $("console").hidden = false;
    const run = async () => {
      const code = $("code").value;
      if (!code.trim()) return;
      $("runmsg").textContent = "running…";
      try {
        let j = await post("/api/v1/debug/exec", { code, wait: 10 });
        while (j.state !== "done") {             // queued behind a long step: poll
          await new Promise((r) => setTimeout(r, 500));
          const r = await fetch(`/api/v1/debug/job/${j.id}`, { headers: { "Authorization": `Bearer ${token()}` } });
          j = await r.json();
        }
        const parts = [`>>> ${code}`];
        if (j.output) parts.push(j.output.replace(/\n$/, ""));
        if (j.value !== null && j.value !== undefined) parts.push(j.value);
        if (j.error) parts.push(j.error);
        $("out").textContent = parts.join("\n") + "\n" + $("out").textContent;
        $("runmsg").textContent = `${j.ok ? "ok" : "error"} (${j.seconds} s, t=${j.t ?? "–"})`;
      } catch (e) { $("runmsg").textContent = e.message; }
    };
    $("run").onclick = run;
    $("code").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } });
  }
}

function show(name) {
  const img = $("img");
  if (Viewer.stream) Viewer.transports[Viewer.transport].detach(img);
  Viewer.stream = name;
  for (const b of $("streams").children) b.classList.toggle("on", b.textContent === name);
  Viewer.transports[Viewer.transport].attach(img, name);
  if (typeof Camera !== "undefined" && $("camera") && !$("camera").hidden) Camera.attach(name).catch(() => {});
}

function setState(s) {
  const p = $("state");
  p.textContent = s;
  p.className = "pill " + s;
}

function renderStatus(snap) {
  const s = snap.status || {};
  setState(s.paused && s.state === "running" ? "paused" : (s.state || "?"));
  const st = (snap.streams || {})[Viewer.stream] || {};
  const rows = [
    ["sim time", s.t !== undefined ? `${fmt(s.t)} s` + (s.t_end ? ` / ${fmt(s.t_end)} s` : "") : "–"],
    ["real-time ×", fmt(s.rtf, 2)],
    ["segment", s.segment || "–"],
    ["kind", s.segment_kind || "–"],
    ["frames/s", fmt(st.fps, 1)],
    ["render share", s.render_share !== undefined && s.render_share !== null ? `${fmt(100 * s.render_share, 0)} %` : "–"],
    ["wall", s.wall_s !== undefined ? `${fmt(s.wall_s, 0)} s` : "–"],
    ["paused", s.paused ? (s.step_until ? `stepping to ${fmt(s.step_until, 2)} s` : "yes") : "no"],
    ["unchanged frames", s.frames ? `${s.unchanged_frames ?? 0} / ${s.frames}` : "–"],
    ["blank frames", s.blank_frames ? `${s.blank_frames} (renderer: see docs/live_view.md)` : "0"],
  ];
  const dl = $("status");
  dl.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    if (k === "blank frames" && v !== "0") dd.className = "fail";
    return [dt, dd];
  }));
  if (s.transfers) renderResult(s);
  renderRecording(s);
}

function renderResult(s) {
  const box = $("result");
  box.replaceChildren();
  const head = document.createElement("div");
  head.className = s.ok ? "ok" : "fail";
  head.textContent = `${s.ok ? "PASS" : "FAIL"}: ${s.passed}/${s.transfers.length} transfers`;
  box.appendChild(head);
  for (const t of s.transfers) {
    const d = document.createElement("div");
    d.className = t.ok ? "ok" : "fail";
    d.textContent = `${t.ok ? "✓" : "✗"} ${t.src} → ${t.dst}`;
    d.title = JSON.stringify(t);
    box.appendChild(d);
  }
}

function addEvent(kind, e) {
  const ol = $("events");
  const li = document.createElement("li");
  const what = kind === "segment" ? e.name : Object.entries(e).filter(([k]) => !["seq", "kind", "wall", "transfers"].includes(k))
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");
  li.textContent = `${new Date(e.wall * 1000).toLocaleTimeString()} ${kind} ${what}`;
  ol.prepend(li);
  while (ol.children.length > 200) ol.lastChild.remove();
}

function listen() {
  const es = new EventSource(`/api/v1/stream?since=${Viewer.events}`);
  es.addEventListener("status", (m) => {
    const snap = JSON.parse(m.data);
    renderStatus(snap);
    if (typeof Camera !== "undefined") Camera.onStatus((snap.status || {}).views);
  });
  es.addEventListener("event", (m) => {
    const e = JSON.parse(m.data);
    Viewer.events = e.seq;
    addEvent(e.kind, e);
    if (e.kind === "record_saved" || e.kind === "record_discarded") refreshRecordings();
  });
  es.onerror = () => { if (!["done", "failed", "stopped", "interrupted"].includes($("state").textContent)) setState("disconnected"); };
}

init().catch((e) => { setState("error"); $("run").textContent = String(e); });

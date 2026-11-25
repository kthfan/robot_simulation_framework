// rsf live viewer: frames from /api/v1/video/<name>/<transport>, status and events
// from /api/v1/stream (SSE). Transports are a table: add one (e.g. webrtc) with
// Viewer.transports.name = {attach(img, stream), detach(img)} and list it in the
// server's /api/v1/info "transports".
"use strict";
const Viewer = {
  transports: {
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
  stream: null, transport: null, events: 0,
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined) ? "–" : (typeof v === "number" ? v.toFixed(d) : String(v));

async function init() {
  const params = new URLSearchParams(location.search);
  const info = await (await fetch("/api/v1/info")).json();
  const i = info.info || {};
  $("run").textContent = [i.backend, i.layout, i.plan].filter(Boolean).join(" · ");
  const names = Object.keys(info.streams);
  const want = params.get("transport");
  Viewer.transport = (want && Viewer.transports[want]) ? want
                   : info.transports.find((t) => Viewer.transports[t]) || "jpeg";
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
  if (names.length) show(params.get("stream") && names.includes(params.get("stream")) ? params.get("stream") : names[0]);
  listen();
}

function show(name) {
  const img = $("img");
  if (Viewer.stream) Viewer.transports[Viewer.transport].detach(img);
  Viewer.stream = name;
  for (const b of $("streams").children) b.classList.toggle("on", b.textContent === name);
  Viewer.transports[Viewer.transport].attach(img, name);
}

function setState(s) {
  const p = $("state");
  p.textContent = s;
  p.className = "pill " + s;
}

function renderStatus(snap) {
  const s = snap.status || {};
  setState(s.state || "?");
  const st = (snap.streams || {})[Viewer.stream] || {};
  const rows = [
    ["sim time", s.t !== undefined ? `${fmt(s.t)} s` + (s.t_end ? ` / ${fmt(s.t_end)} s` : "") : "–"],
    ["real-time ×", fmt(s.rtf, 2)],
    ["segment", s.segment || "–"],
    ["kind", s.segment_kind || "–"],
    ["frames/s", fmt(st.fps, 1)],
    ["render share", s.render_share !== undefined && s.render_share !== null ? `${fmt(100 * s.render_share, 0)} %` : "–"],
    ["wall", s.wall_s !== undefined ? `${fmt(s.wall_s, 0)} s` : "–"],
  ];
  const dl = $("status");
  dl.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    return [dt, dd];
  }));
  if (s.transfers) renderResult(s);
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
  es.addEventListener("status", (m) => renderStatus(JSON.parse(m.data)));
  es.addEventListener("event", (m) => { const e = JSON.parse(m.data); Viewer.events = e.seq; addEvent(e.kind, e); });
  es.onerror = () => { if (!["done", "failed"].includes($("state").textContent)) setState("disconnected"); };
}

init().catch((e) => { setState("error"); $("run").textContent = String(e); });

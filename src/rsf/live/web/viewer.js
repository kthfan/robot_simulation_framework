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
  Viewer.info = i;
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
  setupGated(info.features || {});
  listen();
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
  es.onerror = () => { if (!["done", "failed", "stopped", "interrupted"].includes($("state").textContent)) setState("disconnected"); };
}

init().catch((e) => { setState("error"); $("run").textContent = String(e); });

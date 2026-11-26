// Camera control of the shown view: keyboard (WASD, T/G: in the camera's own frame), mouse (drag to look, FPS mouse
// look on double-click, right/middle/Shift drag to pan, wheel to move, Ctrl+wheel to zoom).
// The camera is the server's (shared: every viewer sees it where the last one put it);
// this computes the pose from the input and sends it (absolute, at most ~30 per second),
// and follows the server's pose (status "views") when someone else moves it.
"use strict";
const Camera = (() => {
  const S = {
    view: null, pose: null, frame: null,
    keys: new Set(), drag: null, last: 0,
    dirty: false, inflight: false, quietUntil: 0,
    set: { speed: 0.5, sens: 0.15, invertY: false },
  };
  const KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyT", "KeyG"]);
  const rad = (d) => d * Math.PI / 180;
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const $ = (id) => document.getElementById(id);

  try { Object.assign(S.set, JSON.parse(localStorage.getItem("rsf-live-camera") || "{}")); } catch (e) { /* none */ }
  delete S.set.fly;                                    // (an earlier option: moving is always in the camera's frame)
  const save = () => { try { localStorage.setItem("rsf-live-camera", JSON.stringify(S.set)); } catch (e) { /* */ } };

  // yaw about world z from +x (counter-clockwise from above), pitch up from horizontal (server's convention)
  function basis(p) {
    const y = rad(p.yaw), q = rad(p.pitch);
    const f = [Math.cos(q) * Math.cos(y), Math.cos(q) * Math.sin(y), Math.sin(q)];     // view direction
    const r = [Math.sin(y), -Math.cos(y), 0];                                          // right
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];  // camera up
    return { f, r, u };
  }
  const add = (e, v, k) => { e[0] += v[0] * k; e[1] += v[1] * k; e[2] += v[2] * k; };

  function changed() {
    S.dirty = true;
    S.quietUntil = performance.now() + 800;            // own moves win over the server's echo
    readout();
    send();
  }

  async function send() {
    if (!S.dirty || S.inflight || !S.view) return;
    S.inflight = true;
    S.dirty = false;
    const p = S.pose;
    try {
      await fetch(`/api/v1/views/${encodeURIComponent(S.view)}/pose`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eye: p.eye, yaw: p.yaw, pitch: p.pitch, fov_deg: p.fov_deg }),
      });
    } catch (e) { /* the next move resends */ }
    setTimeout(() => { S.inflight = false; send(); }, 33);
  }

  function tick(now) {
    const dt = Math.min(0.1, (now - (S.last || now)) / 1000);
    S.last = now;
    if (S.pose && S.keys.size) {
      const b = basis(S.pose);
      let v = S.set.speed * dt;
      if (S.keys.has("Shift")) v *= 4;
      if (S.keys.has("Alt")) v *= 0.25;
      const e = S.pose.eye;
      let moved = false;
      const go = (code, dir, k) => { if (S.keys.has(code)) { add(e, dir, k); moved = true; } };
      // the camera's own frame: W/S along the view (pitch included: looking down, W goes down),
      // A/D to its right, T/G to its up (tilted with the view)
      go("KeyW", b.f, v); go("KeyS", b.f, -v);
      go("KeyD", b.r, v); go("KeyA", b.r, -v);
      go("KeyT", b.u, v); go("KeyG", b.u, -v);
      if (moved) changed();
    }
    requestAnimationFrame(tick);
  }

  function look(dx, dy) {
    const p = S.pose;
    p.yaw = ((p.yaw - dx * S.set.sens) % 360 + 540) % 360 - 180;          // mouse right: turn right
    p.pitch = clamp(p.pitch - dy * S.set.sens * (S.set.invertY ? -1 : 1), -89.9, 89.9);   // mouse up: look up
    changed();
  }

  function pan(dx, dy) {                               // grab the scene: it follows the pointer
    const b = basis(S.pose), k = S.set.speed * 0.004;
    add(S.pose.eye, b.r, -dx * k);
    add(S.pose.eye, b.u, dy * k);
    changed();
  }

  function readout() {
    const p = S.pose;
    if (!p) return;
    const f = (x) => x.toFixed(3);
    $("campose").textContent = `eye (${p.eye.map(f).join(", ")}) · yaw ${p.yaw.toFixed(1)}° · pitch ${p.pitch.toFixed(1)}° · fov ${p.fov_deg.toFixed(0)}°`;
    $("camfov").value = p.fov_deg;
  }

  function adopt(p) {
    S.pose = { eye: p.eye.slice(), yaw: p.yaw, pitch: p.pitch, fov_deg: p.fov_deg };
    readout();
  }

  function onStatus(views) {                           // someone (maybe us) moved a camera
    if (!views || !S.view || !views[S.view]) return;
    if (S.inflight || S.dirty || performance.now() < S.quietUntil || S.drag || S.keys.size) return;
    adopt(views[S.view]);
  }

  async function attach(view) {
    S.view = view;
    const all = await (await fetch("/api/v1/views")).json();
    if (all[view]) adopt(all[view]);
  }

  function bindSettings() {
    const sp = $("camspeed"), spv = $("camspeedv");
    const showSpeed = () => { sp.value = Math.log10(S.set.speed); spv.textContent = `${S.set.speed.toFixed(2)} m/s`; };
    sp.oninput = () => { S.set.speed = Math.pow(10, Number(sp.value)); showSpeed(); save(); };
    showSpeed();
    S.showSpeed = showSpeed;
    $("camsens").value = S.set.sens;
    $("camsens").oninput = () => { S.set.sens = Number($("camsens").value); save(); };
    $("caminv").checked = S.set.invertY;
    $("caminv").onchange = () => { S.set.invertY = $("caminv").checked; save(); };
    $("camfov").oninput = () => { if (S.pose) { S.pose.fov_deg = Number($("camfov").value); changed(); } };
    $("camreset").onclick = reset;
    $("camcopy").onclick = async () => {
      const p = S.pose, b = basis(p);
      const spec = { name: S.view, eye: p.eye.map((x) => +x.toFixed(4)),
        target: p.eye.map((x, i) => +(x + b.f[i]).toFixed(4)), fov_deg: +p.fov_deg.toFixed(1) };
      const text = `--live-view '${JSON.stringify(spec)}'`;
      try { await navigator.clipboard.writeText(text); $("camcopy").textContent = "Copied"; }
      catch (e) { window.prompt("Copy:", text); }
      setTimeout(() => { $("camcopy").textContent = "Copy as --live-view"; }, 1500);
    };
    $("camhelp").onclick = () => { $("help").hidden = !$("help").hidden; };
  }

  async function reset() {
    if (!S.view) return;
    S.dirty = false;
    S.quietUntil = 0;
    const r = await fetch(`/api/v1/views/${encodeURIComponent(S.view)}/reset`, { method: "POST" });
    const p = await r.json();
    const d = p.target.map((x, i) => x - p.eye[i]), n = Math.hypot(...d);
    adopt({ eye: p.eye, yaw: Math.atan2(d[1], d[0]) * 180 / Math.PI, pitch: Math.asin(d[2] / n) * 180 / Math.PI,
            fov_deg: p.fov_deg });
  }

  function bindInput(frame) {
    S.frame = frame;
    frame.tabIndex = 0;
    frame.addEventListener("keydown", (e) => {
      if (e.code === "AltLeft" || e.code === "AltRight") { S.keys.add("Alt"); e.preventDefault(); return; }
      if (e.key === "Shift") { S.keys.add("Shift"); return; }
      if (KEYS.has(e.code)) { S.keys.add(e.code); e.preventDefault(); return; }
      if (e.code === "BracketRight") { S.set.speed = Math.min(5, S.set.speed * 1.5); S.showSpeed(); save(); }
      else if (e.code === "BracketLeft") { S.set.speed = Math.max(0.05, S.set.speed / 1.5); S.showSpeed(); save(); }
      else if (e.code === "KeyR") reset();
      else if (e.key === "?") $("help").hidden = !$("help").hidden;
      else return;
      e.preventDefault();
    });
    frame.addEventListener("keyup", (e) => {
      if (e.code === "AltLeft" || e.code === "AltRight") S.keys.delete("Alt");
      else if (e.key === "Shift") S.keys.delete("Shift");
      else S.keys.delete(e.code);
    });
    frame.addEventListener("blur", () => S.keys.clear());
    frame.addEventListener("contextmenu", (e) => e.preventDefault());
    frame.addEventListener("mousedown", (e) => {
      frame.focus();
      if (!S.pose || document.pointerLockElement === frame) return;
      S.drag = (e.button === 0 && !e.shiftKey) ? "look" : "pan";
      e.preventDefault();
    });
    window.addEventListener("mouseup", () => { S.drag = null; });
    window.addEventListener("mousemove", (e) => {
      if (!S.pose) return;
      if (document.pointerLockElement === frame) look(e.movementX, e.movementY);
      else if (S.drag === "look") look(e.movementX, e.movementY);
      else if (S.drag === "pan") pan(e.movementX, e.movementY);
    });
    const help = document.getElementById("help");
    for (const ev of ["mousedown", "dblclick", "wheel"]) help.addEventListener(ev, (e) => e.stopPropagation());
    frame.addEventListener("dblclick", () => { if (frame.requestPointerLock) frame.requestPointerLock(); });
    document.addEventListener("pointerlockchange", () => {
      frame.classList.toggle("locked", document.pointerLockElement === frame);
    });
    frame.addEventListener("wheel", (e) => {
      if (!S.pose) return;
      e.preventDefault();
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;            // lines -> pixels
      if (e.ctrlKey) {                                                     // zoom (also a trackpad pinch)
        S.pose.fov_deg = clamp(S.pose.fov_deg * Math.exp(dy * 0.002), 5, 120);
      } else {                                                             // move along the view
        add(S.pose.eye, basis(S.pose).f, -dy / 100 * S.set.speed * 0.3);
      }
      changed();
    }, { passive: false });
  }

  return {
    init(frame) { bindInput(frame); bindSettings(); requestAnimationFrame(tick); },
    attach, onStatus, reset,
    get pose() { return S.pose; },
  };
})();

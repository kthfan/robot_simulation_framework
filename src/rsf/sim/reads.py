"""Marker reads during a simulated run (P6): the camera the plan holds still for renders
what it sees, a detector finds the printed markers, and the reads locate the fixtures
they are printed on.

A cell says what its reads are through one hook, named by ``process.reads`` (a
``vision`` registry entry, e.g. ``tm_tray.reads``)::

    hook(world) -> ReadSetup

* ``cameras``: :class:`ReadCamera` by name (a pinhole camera fixed to a robot link,
  given as OpenCV does: optical frame z forward, x right, y down);
* ``markers``: :class:`Marker` (a printed sheet: its pose and size in the world, the
  grey image, and ``board_T``: the pose the detector reports for it);
* ``match(segment_name)``: for a plan's hold segment, ``(feature key, read, camera)``
  when it is a read, else None;
* ``detect(grey, fx, fy, cx, cy)``: ``{board_id: T_camera_board}`` found in an image;
* ``fit_reads``: the reads that locate a fixture (e.g. the near one), and
  ``board_points``: board-frame points (4 x n, homogeneous) the fit matches.

:class:`Reader` is called at every simulated sample (``cell.run(on_sample=...)``); at
the middle of each read's hold it asks the backend for the camera image
(``cell.camera_image(name) -> (grey, (fx, fy, cx, cy))``: the image and the
intrinsics it was rendered with) and for the link's pose (``cell.link_pose(link)``),
as the real cell would compute the camera's pose from its joints. A read's board
pose in the world is compared with the simulated cell's truth.

:func:`fit_entities` turns the reads into each fixture's displacement on the floor
(yaw about the vertical, then x, y, z) against the layout the robot was taught in,
and :func:`located_layout` writes that layout with the fixtures displaced
(``displace: {matrix}``): the one to teach and plan the rest of the run in.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class ReadCamera:
    name: str
    link: str
    T_link_cv: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Marker:
    key: str                  #: the feature it identifies
    entity: str               #: what it is printed on (the fixture a fit moves)
    T_sheet: np.ndarray       #: sheet centre: x right, y up as read, z toward the reader
    size: tuple               #: (width, height) m
    image: np.ndarray         #: grey, rows top to bottom
    board_T: np.ndarray       #: the detector's board frame in the world
    board_id: int


@dataclass
class ReadSetup:
    cameras: dict
    markers: list
    match: Callable
    detect: Optional[Callable] = None
    fit_reads: tuple = ()
    board_points: Optional[np.ndarray] = None


def setup_for(world) -> Optional[ReadSetup]:
    """The world's read setup (``process.reads``), or None when it has none."""
    name = (world.process or {}).get("reads")
    if not name:
        return None
    from rsf.model import registry
    return registry.get("vision", name)(world)


def schedule(setup: ReadSetup, segments) -> list:
    """``[{t, segment, key, read, camera}]``: every read of a plan, at the middle of its hold."""
    out = []
    for s in segments:
        if s.get("kind") != "hold":
            continue
        m = setup.match(s["name"])
        if m:
            key, read, cam = m
            out.append({"t": 0.5 * (float(s["t0"]) + float(s["t1"])), "segment": s["name"], "key": key,
                        "read": read, "camera": cam})
    return sorted(out, key=lambda r: r["t"])


def _pose_error(a, b) -> tuple:
    c = max(-1.0, min(1.0, (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1e3, math.degrees(math.acos(c))


class Reader:
    """Takes the plan's reads as the run passes them (``__call__(t, *_)`` per sample)."""

    def __init__(self, setup: ReadSetup, segments, cell, images: Optional[str] = None):
        self.setup, self.cell = setup, cell
        self.todo = schedule(setup, segments)
        self.marker = {m.key: m for m in setup.markers}
        self.images = Path(images) if images else None
        self.rows = []

    def __call__(self, t, *_):
        while self.todo and t >= self.todo[0]["t"] - 1e-9:
            self.rows.append(self._read(self.todo.pop(0), t))

    def _read(self, r: dict, t: float) -> dict:
        cam = self.setup.cameras[r["camera"]]
        img, (fx, fy, cx, cy) = self.cell.camera_image(cam.name)
        T_cv = np.asarray(self.cell.link_pose(cam.link), float) @ np.asarray(cam.T_link_cv, float)
        mk = self.marker[r["key"]]
        rec = {"slot": r["key"], "entity": mk.entity, "read": r["read"], "segment": r["segment"],
               "t": round(float(t), 3), "camera": cam.name, "board_id": mk.board_id,
               "T_cv": np.round(T_cv, 9).tolist(), "K": [float(fx), float(fy), float(cx), float(cy)]}
        if self.images is not None:
            from PIL import Image
            self.images.mkdir(parents=True, exist_ok=True)
            rec["image"] = image_name(rec)
            Image.fromarray(img).save(self.images / rec["image"])
        if self.setup.detect is None:
            rec["found"] = None
            return rec
        return detect_row(self.setup, rec, img, mk)

    def summary(self) -> dict:
        return summarize(self.rows, len(self.todo))


def image_name(rec: dict) -> str:
    return f"{rec['slot'].replace('/', '_')}_{rec['read']}.png"


def detect_row(setup: ReadSetup, rec: dict, img, mk: Marker) -> dict:
    """*rec* with what the detector finds in *img* (taken at ``rec["T_cv"]`` with ``rec["K"]``)."""
    fx, fy, cx, cy = rec["K"]
    T_cv = np.asarray(rec["T_cv"], float)
    boards = setup.detect(img, fx, fy, cx, cy)
    rec["boards_seen"] = sorted(int(b) for b in boards)
    rec["found"] = mk.board_id in boards
    if rec["found"]:
        B = T_cv @ boards[mk.board_id]
        rec["board_T"] = np.round(B, 9).tolist()
        rec["pos_err_mm"], rec["rot_err_deg"] = (round(v, 4) for v in _pose_error(B, mk.board_T))
        rec["distance_m"] = round(float(np.linalg.norm(boards[mk.board_id][:3, 3])), 4)
    return rec


def detect_saved(setup: ReadSetup, rows: list, images) -> list:
    """The rows of a run whose images were saved without a detector, detected now."""
    from PIL import Image
    marker = {m.key: m for m in setup.markers}
    return [detect_row(setup, dict(r), np.asarray(Image.open(Path(images) / r["image"]).convert("L")), marker[r["slot"]])
            for r in rows]


def summarize(rows: list, missed: int = 0) -> dict:
    found = [x for x in rows if x.get("found")]
    out = {"reads": len(rows), "found": len(found), "missed": missed}
    if found:
        for k in ("pos_err_mm", "rot_err_deg"):
            v = [x[k] for x in found]
            out[k] = {"mean": round(float(np.mean(v)), 4), "max": round(float(max(v)), 4)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Locating fixtures from the reads
# ─────────────────────────────────────────────────────────────────────────────
def fit_planar(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The world transform (yaw about z, then x, y, z) taking points *P* (3 x n) to *Q*
    in the least-squares sense."""
    p0, q0 = P.mean(axis=1, keepdims=True), Q.mean(axis=1, keepdims=True)
    a, b = P[:2] - p0[:2], Q[:2] - q0[:2]
    th = math.atan2(float(np.sum(a[0] * b[1] - a[1] * b[0])), float(np.sum(a[0] * b[0] + a[1] * b[1])))
    R = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (q0 - R @ p0).ravel()
    return T


def planar(B_est: np.ndarray, B_nom: np.ndarray) -> np.ndarray:
    """The world correction taking the nominal board *B_nom* to the read *B_est*,
    for something standing on the floor: only its yaw, and the translation that
    puts the board's origin where it was read (the read's small roll and pitch
    turned about the world origin would move the board by millimetres)."""
    C = B_est @ np.linalg.inv(B_nom)
    th = math.atan2(C[1, 0], C[0, 0])
    T = np.eye(4)
    T[:3, :3] = [[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = B_est[:3, 3] - T[:3, :3] @ B_nom[:3, 3]
    return T


def describe(D: np.ndarray, about) -> dict:
    """``D`` as yaw (deg) about *about* and the translation that follows (mm)."""
    th = math.atan2(D[1, 0], D[0, 0])
    c = np.asarray(about, float)
    t = D[:3, 3] - (c - D[:3, :3] @ c)
    return {"yaw_deg": round(math.degrees(th), 4), "xyz_mm": [round(float(v) * 1e3, 3) for v in t]}


def fit_entities(setup_nominal: ReadSetup, rows: list, entities=None) -> dict:
    """``{entity: {D, reads, estimated}}``: each fixture's floor displacement from the
    nominal layout (*setup_nominal*: its markers where the robot was taught) to where
    the fit reads (``fit_reads``) found them. *entities*: only these."""
    nom = {m.key: m for m in setup_nominal.markers}
    pts = setup_nominal.board_points
    groups = {}
    for r in rows:
        if not r.get("found") or r["read"] not in setup_nominal.fit_reads or r["slot"] not in nom:
            continue
        m = nom[r["slot"]]
        if entities is not None and m.entity not in entities:
            continue
        P, Q = groups.setdefault(m.entity, ([], []))
        P.append((m.board_T @ pts)[:3])
        Q.append((np.asarray(r["board_T"], float) @ pts)[:3])
    out = {}
    for ent, (P, Q) in groups.items():
        D = fit_planar(np.hstack(P), np.hstack(Q))
        about = np.mean([nom[r["slot"]].board_T[:3, 3] for r in rows if r.get("found") and r["slot"] in nom
                         and nom[r["slot"]].entity == ent], axis=0)
        out[ent] = {"D": D, "reads": len(P), "about": about, "estimated": describe(D, about)}
    return out


def located_layout(nominal_path, out_path, name: str, fits: dict, note: str = "") -> Path:
    """The nominal layout with each fitted fixture displaced (``displace: {matrix}``)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(Path(nominal_path).resolve(), out.parent.resolve())
    lines = [f"# Written by rsf.sim.reads{': ' + note if note else ''}: the nominal layout with the",
             "# fixtures where the marker reads put them. Teach and plan in this layout.",
             "schema: rsf.layout/1", f"extends: {rel}", f"name: {name}_located", "entities:"]
    for ent, fit in fits.items():
        mat = [[round(float(v), 9) for v in row] for row in np.asarray(fit["D"])]
        lines.append(f"  {ent}:")
        lines.append(f"    displace: {{matrix: {json.dumps(mat)}}}   # {fit['estimated']}")
    out.write_text("\n".join(lines) + "\n")
    return out

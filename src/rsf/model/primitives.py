"""Build simple collision geometry (boxes) as mesh files and URDF links.

Convex pieces are kept separate on purpose: MuJoCo and most CPU collision
libraries treat each mesh as its convex hull, so a concave part (a tray with a
recessed underside) must be written as several convex boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from rsf.model.urdf import Geometry, Link

_FACES = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]


@dataclass(frozen=True)
class BoxSpec:
    """An axis-aligned box ``[lo, hi]`` in the part frame."""
    name: str
    lo: tuple
    hi: tuple

    @classmethod
    def from_dict(cls, d: dict) -> "BoxSpec":
        if "lo" in d:
            return cls(d["name"], tuple(map(float, d["lo"])), tuple(map(float, d["hi"])))
        c, s = np.asarray(d["center"], float), np.asarray(d["size"], float)
        return cls(d["name"], tuple(c - s / 2), tuple(c + s / 2))


def write_box_obj(path: Path, lo, hi, comment: str = "") -> Path:
    """A closed, outward-facing box as a 12-triangle OBJ."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.any(hi <= lo):
        raise ValueError(f"empty box {lo} .. {hi}")
    v = [(x, y, z) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for p in v:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*p))
        for a, b, c, d in _FACES:
            f.write(f"f {a} {b} {c}\nf {a} {c} {d}\n")
    return path


def box_link(name: str, boxes: Iterable[BoxSpec], mesh_dir: Path, mass: Optional[float] = None) -> Link:
    """A link whose visuals and collisions are one OBJ box per :class:`BoxSpec`."""
    geos = []
    for b in boxes:
        p = write_box_obj(Path(mesh_dir) / f"{b.name}.obj", b.lo, b.hi, f"{name}/{b.name}")
        geos.append(Geometry("mesh", np.eye(4), name=b.name, filename=str(p), path=p.resolve()))
    return Link(name, visuals=list(geos), collisions=list(geos), mass=mass)


def recessed_tray(width: float, length: float, height: float, recess: float, rail: float) -> list:
    """A tray on two rails along its long (y) edges, underside recessed between them.

    Frame: origin at the centre of the rails' underside, x across, y along, z up.
    """
    x, y = width / 2.0, length / 2.0
    if recess <= 0.0 or rail <= 0.0:
        return [BoxSpec("body", (-x, -y, 0.0), (x, y, height))]
    return [
        BoxSpec("rail_left", (-x, -y, 0.0), (-x + rail, y, height)),
        BoxSpec("rail_right", (x - rail, -y, 0.0), (x, y, height)),
        BoxSpec("body", (-x + rail, -y, recess), (x - rail, y, height)),
    ]

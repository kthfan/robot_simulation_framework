"""Render a robot's end effector (collision geometry) in its tool frame, four views.

    PYTHONPATH=src python tools/render_end_effector.py cells/tm_tray/assets/tm5s_egu50.urdf \
        --frame flange --links link_6 gripper_body right_gripper left_gripper \
        --joint right_gripper_joint=0.057675 --joint left_gripper_joint=0.057675 --out ee.png

Headless (matplotlib). Every collision geometry of the listed links gets its own
colour and a legend entry, with its extent in the tool frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsf.model.urdf import UrdfModel  # noqa: E402

PALETTE = ["#9aa5b1", "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
           "#64748b", "#a3e635"]


def parts(model: UrdfModel, frame: str, links, q: dict):
    import trimesh
    fk = model.fk(q, base=frame)
    out = []
    for ln in links:
        for g in model.links[ln].collisions:
            m = trimesh.load(g.path, force="mesh")
            V = (fk[ln] @ g.origin @ np.c_[m.vertices, np.ones(len(m.vertices))].T).T[:, :3]
            out.append((f"{ln}/{g.name}" if g.name != ln else ln, V, np.asarray(m.faces)))
    return out


def render(parts_, out: Path, title: str) -> None:
    """One isometric 3-D view and three true orthographic projections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    allV = np.vstack([V for _, V, _ in parts_])
    lo, hi = allV.min(0), allV.max(0)
    fig = plt.figure(figsize=(17, 15))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for i, (pname, V, F) in enumerate(parts_):
        ax.add_collection3d(Poly3DCollection(V[F][:, :, [2, 0, 1]], facecolor=PALETTE[i % len(PALETTE)],
                                             edgecolor=(0, 0, 0, 0.15 if len(F) < 50 else 0.0), linewidths=0.3,
                                             alpha=0.5 if pname.startswith("link") else 0.85))
    c, r = (lo + hi)[[2, 0, 1]] / 2, (hi - lo).max() / 2 * 1.02
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("z approach (m)"); ax.set_ylabel("x lateral (m)"); ax.set_zlabel("y up (m)")
    ax.set_title("isometric")
    views = [("side, seen from +x: z (approach) right, y up", 2, 1, False),
             ("top, seen from +y: z (approach) right, x (lateral) up", 2, 0, True),
             ("front, seen from +z (into the tool): x right, y up", 0, 1, False)]
    for k, (name, h, v, flip_draw) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 2)
        # draw far parts first so near ones stay visible
        depth_axis = ({0, 1, 2} - {h, v}).pop()
        order = sorted(range(len(parts_)), key=lambda i: parts_[i][1][:, depth_axis].mean(), reverse=not flip_draw)
        for i in order:
            pname, V, F = parts_[i]
            ax.add_collection(PolyCollection(V[F][:, :, [h, v]], facecolor=PALETTE[i % len(PALETTE)],
                                             edgecolor="none", alpha=0.45 if pname.startswith("link") else 0.85))
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_xlim(lo[h] - 0.01, hi[h] + 0.01); ax.set_ylim(lo[v] - 0.01, hi[v] + 0.01)
        ax.set_aspect("equal"); ax.grid(True, lw=0.3)
        lab = "xyz"
        ax.set_xlabel(f"{lab[h]} (m)"); ax.set_ylabel(f"{lab[v]} (m)"); ax.set_title(name)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i % len(PALETTE)]) for i in range(len(parts_))]
    labels = [f"{n}: x {V[:, 0].min()*1e3:.1f}..{V[:, 0].max()*1e3:.1f}, y {V[:, 1].min()*1e3:.1f}..{V[:, 1].max()*1e3:.1f}, "
              f"z {V[:, 2].min()*1e3:.1f}..{V[:, 2].max()*1e3:.1f} mm" for n, V, _ in parts_]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out, dpi=110)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--frame", default="flange")
    ap.add_argument("--links", nargs="+", required=True)
    ap.add_argument("--joint", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    q = {k: float(v) for k, v in (j.split("=") for j in a.joint)}
    model = UrdfModel.from_file(a.urdf)
    render(parts(model, a.frame, a.links, q), Path(a.out),
           a.title or f"{Path(a.urdf).name}: collision geometry in the {a.frame} frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())

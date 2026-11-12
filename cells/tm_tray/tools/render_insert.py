"""Side section (flange y-z plane) of the gripper, target jig and tray at a grasp phase.

    PYTHONPATH=src python cells/tm_tray/tools/render_insert.py cells/tm_tray/layouts/strict.yaml \
        --slot incoming/L0/C0 --out docs/images/insert_section.png [--push-margin 0.002]

Every box within ``--band`` of the flange's x = 0 plane is drawn as its y-z
outline; overlaps between the gripper and the target tray are hatched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from rsf.core import se3  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402
from rsf.model.urdf import UrdfModel  # noqa: E402
from rsf.model.world import solids_from_urdf  # noqa: E402

COLORS = {"tray": "#f59e0b", "jig": "#64748b", "plate": "#ef4444", "air_cylinder_plate": "#a855f7",
          "body": "#3b82f6", "jaw": "#10b981", "shelf": "#94a3b8"}


def boxes_in_flange(world, key, T_flange, band):
    inv = se3.inv(T_flange)
    out = []
    for s in world.solids():
        if s.shape.kind != "box":
            continue
        C = (s.corners() - T_flange[:3, 3]) @ T_flange[:3, :3]
        if C[:, 0].max() < -band or C[:, 0].min() > band or C[:, 2].max() < -0.1 or C[:, 2].min() > 0.45:
            continue
        if not (s.name.startswith((f"tray/{key}/", f"jig/{key}/")) or "shelf" in s.name):
            continue
        kind = "tray" if s.name.startswith("tray/") else "jig" if s.name.startswith("jig/") else "shelf"
        out.append((s.name, kind, C))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--slot", default="incoming/L0/C0")
    ap.add_argument("--phase", default="insert")
    ap.add_argument("--push-margin", type=float, nargs="*", default=[None, 0.002],
                    help="panels: None = the current sequence, a number = stop the push plate that far from the tray")
    ap.add_argument("--band", type=float, default=0.045)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    world = load_world(a.layout)
    import rsf_plugins.tm_tray.sequence as SEQ
    from rsf_plugins.tm_tray import hardware as A
    from rsf_plugins.tm_tray.geometry import slot_of
    orig = SEQ.insert_tip_depth
    # flange frame: robot geometry is defined there; the URDF root frame is not the flange
    m = UrdfModel.from_file(world.robot.urdf)
    fk0 = m.fk({"right_gripper_joint": A.JAW_Q_OPEN, "left_gripper_joint": A.JAW_Q_OPEN})
    inv_f = se3.inv(fk0["flange"])
    grip = []
    for s in solids_from_urdf(m, q={"right_gripper_joint": A.JAW_Q_OPEN, "left_gripper_joint": A.JAW_Q_OPEN}):
        link = s.name.split("/")[0]
        if link in ("gripper_body", "right_gripper", "left_gripper") and s.shape.kind == "box":
            C = (inv_f @ np.c_[s.corners(), np.ones(8)].T).T[:, :3]
            nm = s.name.split("/")[-1]
            grip.append((nm, "jaw" if link != "gripper_body" else (nm if nm in COLORS else "body"), C))
    push = next((C for n, k, C in grip if n == "air_cylinder_plate"), None)

    panels = a.push_margin
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 5.2 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, pm in zip(axes, panels):
        if pm is None:
            SEQ.insert_tip_depth = orig
            label = f"current sequence (D28): plate tip {orig() * 1e3:.1f} mm behind the rack face"
        else:
            lim = A.PLATE_FZ_TIP + A.JIG_Y_OFFSET - A.TRAY_Y / 2 - push[:, 2].max() - pm
            SEQ.insert_tip_depth = lambda *x, _l=lim, **k: min(orig(*x, **k), _l)
            label = (f"push plate stopped {pm * 1e3:.0f} mm short of the tray: plate tip "
                     f"{SEQ.insert_tip_depth() * 1e3:.1f} mm behind the rack face")
        T = SEQ.grasp_sequence(world.process, slot_of(world.feature(a.slot)))[a.phase]
        cell = boxes_in_flange(world, a.slot, T, a.band)
        tray_boxes = [C for n, k, C in cell if k == "tray"]
        for name, kind, C in cell + grip:
            if kind != "jaw" and (C[:, 0].max() < -a.band or C[:, 0].min() > a.band):
                continue
            z0, z1, y0, y1 = C[:, 2].min(), C[:, 2].max(), C[:, 1].min(), C[:, 1].max()
            if kind == "jaw":   # outside the section (|x| 83-88 mm): outline only
                ax.add_patch(Polygon([(z0, y0), (z1, y0), (z1, y1), (z0, y1)], closed=True, fill=False,
                                     edgecolor=COLORS["jaw"], lw=1.2, ls="--"))
                continue
            ax.add_patch(Polygon([(z0, y0), (z1, y0), (z1, y1), (z0, y1)], closed=True,
                                 facecolor=COLORS.get(kind, "#cbd5e1"), alpha=0.25 if kind in ("tray", "shelf") else 0.7,
                                 edgecolor=COLORS.get(kind, "#334155"), lw=1.2))
            if kind not in ("tray", "jig", "shelf", "jaw"):
                for TB in tray_boxes:
                    oz = (max(z0, TB[:, 2].min()), min(z1, TB[:, 2].max()))
                    oy = (max(y0, TB[:, 1].min()), min(y1, TB[:, 1].max()))
                    if oz[1] > oz[0] and oy[1] > oy[0]:
                        ax.add_patch(Polygon([(oz[0], oy[0]), (oz[1], oy[0]), (oz[1], oy[1]), (oz[0], oy[1])],
                                             closed=True, fill=False, hatch="xxx", edgecolor="red", lw=1.5))
        if push is not None:
            tray_front = min(TB[:, 2].min() for TB in tray_boxes)
            gap = tray_front - push[:, 2].max()
            ax.annotate(f"push plate -> tray front: {gap * 1e3:+.1f} mm", xy=(push[:, 2].max(), 0.06),
                        xytext=(0.15, 0.085), arrowprops=dict(arrowstyle="->"), fontsize=11)
        ax.set_xlim(-0.01, 0.40); ax.set_ylim(-0.075, 0.10); ax.set_aspect("equal"); ax.grid(True, lw=0.3)
        ax.set_xlabel("flange z (m): approach"); ax.set_ylabel("flange y (m): up")
        ax.set_title(f"{a.slot} at {a.phase} -- {label}", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in COLORS.values()]
    fig.legend(handles, ["tray (rails + recessed body)", "target jig", "plate", "push plate", "gripper body",
                         "jaws (dashed: outside the section)", "shelf"], loc="lower center", ncol=7, frameon=False)
    fig.suptitle(f"Side section |x| < {a.band * 1e3:.0f} mm (flange frame), jaws open; red hatching = overlap", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(a.out, dpi=110)
    SEQ.insert_tip_depth = orig
    return 0


if __name__ == "__main__":
    sys.exit(main())

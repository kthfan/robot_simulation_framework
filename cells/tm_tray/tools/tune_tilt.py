"""Find the tilt motion (pivot, retreat) with the least flange displacement.

The withdraw -> carry move tilts the plate by ``process.carry_tilt_deg`` about a
pivot while backing out (``sequence.TiltMotion``). Candidates on a grid are
tried in order of flange displacement; each is checked on every slot along the
whole motion (continuous IK from every passing withdraw branch, exact meshes at
every sample, D28 margins). The first that passes everywhere is the answer.

    PYTHONPATH=src python cells/tm_tray/tools/tune_tilt.py cells/tm_tray/layouts/strict.yaml --out reports/tilt.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from rsf.model.layout import load_world  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout", nargs="+", help="every layout the answer must serve")
    ap.add_argument("--pivot-y", type=float, nargs="+", default=[-0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08])
    ap.add_argument("--pivot-z", type=float, nargs=3, default=[0.0, 0.38, 0.01], metavar=("FROM", "TO", "STEP"))
    ap.add_argument("--retreat", type=float, nargs=3, default=[0.0, 0.05, 0.005], metavar=("FROM", "TO", "STEP"))
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--allow-forward", action="store_true",
                    help="allow the flange to move forward (deeper into the shelf) during the tilt; "
                         "by default it may only go down and back (user 2026-11-13)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    worlds = [load_world(p) for p in a.layout]
    from rsf.kinematics import load_arm
    import rsf_plugins.tm_tray.sequence as SEQ
    from rsf_plugins.tm_tray.collision_check import CellChecker
    from rsf_plugins.tm_tray.geometry import slot_of
    from rsf_plugins.tm_tray.tool import tool_geometry

    tilt = math.radians(worlds[0].process["carry_tilt_deg"])
    zs = np.arange(a.pivot_z[0], a.pivot_z[1] + 1e-9, a.pivot_z[2])
    rs = np.arange(a.retreat[0], a.retreat[1] + 1e-9, a.retreat[2])
    def forward(m) -> float:
        """Largest forward (approach-axis) flange travel along the motion."""
        return max(m.pose(np.eye(4), tilt, u)[2, 3] for u in np.linspace(0.0, 1.0, m.samples + 1))

    cands = [SEQ.TiltMotion(float(y), float(z), float(r)) for y, z, r in product(a.pivot_y, zs, rs)]
    if not a.allow_forward:
        cands = [m for m in cands if forward(m) <= 1e-9]
    cands.sort(key=lambda m: (round(m.displacement(tilt), 6), m.retreat))
    checkers = [(w, CellChecker(w, load_arm(w))) for w in worlds]
    tried = []
    for m in cands[: a.max_candidates]:
        t = time.time()
        fails, worst = [], np.inf
        for w, cc in checkers:
            proc = dict(w.process, tilt_motion={"pivot": [m.pivot_y, m.pivot_z], "retreat": m.retreat,
                                                "samples": m.samples})
            # hardest first: everything below a shelf; tables and top layers last
            feats = sorted(w.features({"kind": "slot"}), key=lambda f: f.get("role") == "inspection")
            for f in feats:
                phases = SEQ.pick_phases(proc, slot_of(f), tool_geometry(w))
                res = cc._run(f.key, phases[-2:])                  # withdraw, carry (the motion)
                ok = [x for x in res["carry"] if x["ok"]]
                if not ok:
                    why = next((x.get("path") for x in res["carry"] if x.get("path")), "no passing withdraw branch")
                    fails.append((w.name, f.key, why))
                    break
                worst = min(worst, max(x["path_min_slack"] for x in ok))
            if fails:
                break
        end = m.pose(np.eye(4), tilt, 1.0)[:3, 3]
        rec = {"pivot": [m.pivot_y, m.pivot_z], "retreat": m.retreat, "displacement": m.displacement(tilt),
               "flange_down": -float(end[1]), "flange_back": -float(end[2]),
               "ok": not fails, "fail": fails[0] if fails else None,
               "min_slack": None if fails else worst, "seconds": round(time.time() - t, 1)}
        tried.append(rec)
        print(f"pivot y {m.pivot_y:+.3f} z {m.pivot_z:.3f} retreat {m.retreat:.3f}: displacement "
              f"{rec['displacement'] * 1e3:5.1f} mm (down {rec['flange_down'] * 1e3:.1f}, back {rec['flange_back'] * 1e3:.1f})"
              f"  {'PASS' if rec['ok'] else 'fail'}"
              f"{'' if rec['ok'] else '  ' + str(rec['fail'])}  ({rec['seconds']} s)", flush=True)
        if rec["ok"]:
            break
    best = next((r for r in tried if r["ok"]), None)
    print("BEST", json.dumps(best))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"layouts": a.layout, "tilt_deg": math.degrees(tilt),
                                           "best": best, "tried": tried}, indent=1))
    return 0 if best else 1


if __name__ == "__main__":
    sys.exit(main())

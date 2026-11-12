"""M3 collision check of a TM tray layout on the CPU (exact meshes, coal).

    PYTHONPATH=src python cells/tm_tray/tools/check_cell.py cells/tm_tray/layouts/strict.yaml \
        [--robot cells/tm_tray/assets/tm5s_egu50_no_air_cylinder.urdf] [--slots incoming/L0/C0 ...] --out reports/x.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from rsf.model.layout import load_world  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("--robot", help="override the layout's robot URDF")
    ap.add_argument("--slots", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ov = {"robot.urdf": str(Path(a.robot).resolve())} if a.robot else None
    world = load_world(a.layout, ov)
    from rsf_plugins.tm_tray.collision_check import check_cell
    t = time.time()
    rep = check_cell(world, select=a.slots or None)
    rep["seconds"] = round(time.time() - t, 1)
    for s in rep["slots"]:
        print(f"{s['slot']:<22} {'ok  ' if s['ok'] else 'FAIL'} chain={s['chain']} missing={s['missing']}")
    print(f"VERDICT {rep['verdict']} {rep['passed']}/{rep['total']} ({rep['seconds']} s, robot {rep['robot']})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

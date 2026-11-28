"""Detect the reads of a simulated run whose images were saved without a detector
(e.g. Isaac Sim on Kaggle, where ``aprilgrid_detector`` is not installed), and locate
the fixtures from them as ``tools/sim_workflow.py --located`` would.

    PYTHONPATH=src python tools/detect_reads.py cells/tm_tray/layouts/strict.yaml \\
        reports/isaac_reads.json --images reports/isaac_reads_img [--nominal LAYOUT --located OUT.yaml] \\
        --out reports/isaac_reads_detected.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout", help="the simulated layout (the truth the reads are compared with)")
    ap.add_argument("run", help="tools/sim_workflow.py output (its reads.rows)")
    ap.add_argument("--images", required=True, help="the images the run saved (--read-images)")
    ap.add_argument("--nominal", help="the layout the robot was taught in (default: LAYOUT)")
    ap.add_argument("--located", help="write the corrected layout here")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from rsf.model.layout import load_world
    from rsf.sim import reads as RD
    world = load_world(a.layout)
    setup = RD.setup_for(world)
    if setup is None or setup.detect is None:
        print("no read setup or no detector (pip install the aprilgrid_detector wheel)")
        return 2
    rows = RD.detect_saved(setup, json.loads(Path(a.run).read_text())["reads"]["rows"], a.images)
    out = {"layout": a.layout, "run": a.run, "summary": RD.summarize(rows), "rows": rows}
    for r in rows:
        print(f"{r['slot']:16s} {r['read']:10s} " + (f"{r['pos_err_mm']:6.3f} mm {r['rot_err_deg']:6.3f} deg"
                                                    if r.get("found") else f"NOT FOUND (saw {r.get('boards_seen')})"))
    if a.located:
        nom = load_world(a.nominal) if a.nominal else world
        fits = RD.fit_entities(RD.setup_for(nom), rows)
        p = RD.located_layout(a.nominal or a.layout, a.located, nom.name, fits, note="tools/detect_reads.py")
        out["located"] = {"layout": str(p), "fixtures": {k: {"reads": f["reads"], "estimated": f["estimated"]}
                                                         for k, f in fits.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("READS", json.dumps(out["summary"]))
    return 0 if out["summary"]["found"] == out["summary"]["reads"] else 1


if __name__ == "__main__":
    sys.exit(main())

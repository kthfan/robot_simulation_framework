"""Command line: inspect a cell without writing code.

    python -m rsf world  cells/tm_tray/layouts/strict.yaml
    python -m rsf audit  cells/tm_tray/layouts/strict.yaml [--json]
    python -m rsf expand cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml
    python -m rsf scene  cells/tm_tray/layouts/strict.yaml > scene.json     # cuRobo scene dict

``--set dotted.path=value`` (repeatable, YAML value) overrides the layout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import yaml

from rsf.model import audit as AU
from rsf.model.layout import load_world


def _overrides(items) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("world", "audit", "expand", "scene"):
        p = sub.add_parser(name)
        p.add_argument("layout")
        if name == "expand":
            p.add_argument("workflow")
        p.add_argument("--set", action="append", metavar="PATH=VALUE")
        p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    world = load_world(a.layout, _overrides(a.set))

    if a.cmd == "world":
        kinds = Counter(e.kind for e in world.entities)
        feats = Counter(f.kind for f in world.features())
        print(f"{world.name}: {len(world.entities)} entities {dict(kinds)}, {len(world.solids())} solids, "
              f"features {dict(feats)}, {len(world.objects)} objects")
        if world.robot:
            print(f"robot {world.robot.urdf.name} tool {world.robot.tool_frame} base xyz {world.robot.base[:3, 3].round(4).tolist()}")
        return 0
    if a.cmd == "audit":
        checks = AU.audit(world)
        if a.json:
            print(json.dumps({"verdict": AU.verdict(checks), "checks": AU.to_dict(checks)}, indent=1))
        else:
            for c in sorted(checks, key=lambda c: (AU.STATUSES.index(c.status), c.name)):
                print(f"{c.status:4s}  {c.name:38s} {c.detail}")
            print(f"verdict: {AU.verdict(checks)}")
        return 0 if AU.verdict(checks) == AU.PASS else 1
    if a.cmd == "expand":
        from rsf.model.workflow import Workflow, expand
        ex = expand(Workflow.from_file(a.workflow), world)
        if a.json:
            print(json.dumps({"tasks": [t.__dict__ for t in ex.tasks], "end": ex.end.__dict__}, indent=1))
        else:
            for t in ex.tasks:
                print(f"{t.index:4d} c{t.cycle:<3d} {t.skill:10s} {t.args}")
            print(f"end: {ex.end.name} ({ex.end.detail})")
        return 0
    if a.cmd == "scene":
        from rsf.compile.curobo import world_scene
        json.dump(world_scene(world), sys.stdout, indent=1)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""``example.stands``: an audit rule (``python -m rsf audit <layout>``).

A check is ``fn(world) -> list[Check]``; it never needs to raise (the audit turns
an exception into ``skip``). Return ``[]`` when the rule does not apply.
"""
from __future__ import annotations

import itertools

import numpy as np

from rsf.model.audit import check
from rsf.model.registry import register

#: Slots closer than this leave no room for the open fingers beside a neighbour's block.
MIN_SPACING = 0.12


@register("check", "example.stands")
def stands(world) -> list:
    slots = world.features({"kind": "slot"})
    if not slots:
        return []
    out = []
    pts = [(f.key, np.asarray(f.T)[:3, 3]) for f in slots]
    d = min((float(np.linalg.norm(a[1][:2] - b[1][:2])), a[0], b[0]) for a, b in itertools.combinations(pts, 2)) \
        if len(pts) > 1 else (float("inf"), "", "")
    out.append(check("example.stands.spacing", d[0] >= MIN_SPACING,
                     f"slots at least {d[0] * 1e3:.0f} mm apart", f"{d[1]} and {d[2]} only {d[0] * 1e3:.0f} mm apart "
                     f"(needs {MIN_SPACING * 1e3:.0f})", d[0]))
    roles = {f.get("role") for f in slots}
    out.append(check("example.stands.roles", {"source", "target"} <= roles,
                     "source and target slots present", f"roles {sorted(map(str, roles))}: need source and target"))
    return out

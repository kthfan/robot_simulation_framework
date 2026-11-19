"""Shared test setup.

The frozen AI-Robotics-V2 ``cell_core`` (``tests/reference/ai_robotics_v2``) is
the reference the TM tray port is compared against. Its source stays as it was;
corrections the user made since are applied here, once, before any test
imports it, so the comparisons check the port and not the outdated value.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REF = Path(__file__).resolve().parent / "reference" / "ai_robotics_v2"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

#: User corrections to the reference's constants (assets module).
CORRECTIONS = {
    "TRAY_Z": 0.130,          # tray height 130 mm (user 2026-11-19; cell_core: 0.0851)
}


def _apply_reference_corrections() -> None:
    from cell_core import assets as A
    from cell_core import config as C
    for k, v in CORRECTIONS.items():
        setattr(A, k, v)
    A.TRAY_TOP = A.TRAY_BOT + A.TRAY_Z            # derived at import time
    C.MIN_LAYER_GAP = A.TRAY_TOP + A.CORRIDOR_H_FLAT


_apply_reference_corrections()

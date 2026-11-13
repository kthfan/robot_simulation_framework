"""TM5S tray-transfer cell plugin.

Registers the cell's layout generators (``tm_tray.pedestal``, ``tm_tray.rack``,
``tm_tray.table``) and audit checks. Ported from AI-Robotics-V2 ``cell_core``;
the frozen original is in ``tests/reference/ai_robotics_v2``.
"""
from . import generators  # noqa: F401
from . import checks  # noqa: F401
from . import targets  # noqa: F401
from . import skills  # noqa: F401

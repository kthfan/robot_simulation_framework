"""Example cell plugin: the smallest complete cell (docs/new_cell.md).

A plugin is a Python package a layout names under ``plugins:``. Importing it
registers what the cell needs by name (``rsf.model.registry.register``):

* ``generators.py``: ``example.stand`` -- a stand (one box) and the slot on its top;
* ``checks.py``: ``example.stands`` -- an audit rule for the layout;
* ``skills.py``: ``example.pick``, ``example.place``, ``example.transfer`` --
  the motions, as teach-engine sequences.

``geometry.py`` reads the gripper and the block from their URDFs, so the skills
follow the assets when they change.
"""
from . import checks  # noqa: F401
from . import generators  # noqa: F401
from . import skills  # noqa: F401

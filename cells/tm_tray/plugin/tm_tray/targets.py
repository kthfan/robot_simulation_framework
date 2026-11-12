"""Target providers for the TM tray cell: the flange poses that service a slot."""
from __future__ import annotations

from rsf.model.registry import register

from .geometry import slot_of
from .sequence import grasp_sequence
from .tool import tool_geometry


@register("targets", "tm_tray.grasp")
def grasp_targets(world, feature) -> dict:
    """The seven grasp phases of a slot (release visits the same poses)."""
    if feature.kind != "slot":
        return {}
    return grasp_sequence(world.process, slot_of(feature), tool=tool_geometry(world))

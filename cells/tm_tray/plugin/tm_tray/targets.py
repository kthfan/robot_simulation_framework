"""Target providers for the TM tray cell: the flange poses that service a slot."""
from __future__ import annotations

from rsf.model.registry import register

from .geometry import slot_of
from .sequence import pick_phases, place_phases
from .tool import tool_geometry


@register("targets", "tm_tray.grasp")
def grasp_targets(world, feature) -> dict:
    """The pick's poses (without a push plate: cell_core's seven grasp poses)."""
    if feature.kind != "slot":
        return {}
    return {p.name: p.T for p in pick_phases(world.process, slot_of(feature), tool_geometry(world))}


@register("targets", "tm_tray.cycle")
def cycle_targets(world, feature) -> dict:
    """Pick and place: ``pick/<phase>`` and ``place/<phase>``."""
    if feature.kind != "slot":
        return {}
    slot, tool = slot_of(feature), tool_geometry(world)
    out = {f"pick/{p.name}": p.T for p in pick_phases(world.process, slot, tool)}
    out.update({f"place/{p.name}": p.T for p in place_phases(world.process, slot, tool)})
    return out

"""Generators every cell can use. Cell-specific ones live in the cell's plugin."""
from __future__ import annotations

import numpy as np

from rsf.model.registry import register
from rsf.model.world import Entity, Feature, Shape, Solid


@register("generator", "rsf.box")
def box(name: str, params: dict, ctx) -> Entity:
    """One box. ``params``: ``size`` [x, y, z], ``pose`` (layout pose), ``tags``,
    ``kind`` (entity kind, default ``static``)."""
    from rsf.model.layout import parse_pose
    unknown = set(params) - {"size", "pose", "tags", "kind"}
    if unknown:
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}")
    T = parse_pose(params.get("pose"), ctx)
    tags = frozenset(params.get("tags", [])) | {name}
    return Entity(name, params.get("kind", "static"), (Solid(name, T, Shape.box(params["size"]), name, tags),))


@register("generator", "rsf.frame")
def frame(name: str, params: dict, ctx) -> Entity:
    """A named frame and nothing else: ``params``: ``pose``, ``kind``, ``attrs``."""
    from rsf.model.layout import parse_pose
    f = Feature(name, parse_pose(params.get("pose"), ctx), params.get("kind", "frame"), name,
                dict(params.get("attrs", {})))
    return Entity(name, "frame", (), (f,))

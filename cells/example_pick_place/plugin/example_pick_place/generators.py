"""``example.stand``: a stand and the slot on its top.

A generator builds one layout entity: solids (what the robot must not hit) and
features (named frames that workflows and skills select). Its signature is
``fn(name, params, ctx) -> Entity``; *ctx* (``rsf.model.layout.BuildContext``)
gives the layout document and the entities built before this one.

::

    entities:
      stand_a: {generator: example.stand, params: {xy: [0.45, -0.25], yaw_deg: 0, role: source}}

builds the solid ``stand_a`` (a box from the floor up to ``height``) and the
feature ``stand_a/top`` (kind ``slot``, attributes ``role`` and ``stand``): its
origin on the top face's centre, z up, x the direction the gripper's fingers close.
"""
from __future__ import annotations

import math

from rsf.core import se3
from rsf.model.registry import register
from rsf.model.world import Entity, Feature, Shape, Solid

DEFAULTS = {"size": [0.10, 0.10], "height": 0.15, "yaw_deg": 0.0, "role": ""}


@register("generator", "example.stand")
def stand(name: str, params: dict, ctx) -> Entity:
    unknown = set(params) - set(DEFAULTS) - {"xy"}
    if unknown:                                   # a typo must not pass silently
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    x, y = map(float, p["xy"])
    h = float(p["height"])
    R = se3.rot_z(math.radians(float(p["yaw_deg"])))
    box = Solid(name, se3.make_T(R, (x, y, h / 2)), Shape.box([*p["size"], h]), name,
                frozenset({name, "stand"}))
    top = Feature(f"{name}/top", se3.make_T(R, (x, y, h)), "slot", name,
                  {"role": p["role"], "stand": name, "height": h})
    return Entity(name, "static", (box,), (top,))

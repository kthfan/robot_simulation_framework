"""The example cell's skills: pick, place and transfer a block, as teach sequences.

A **skill** is ``fn(world, args) -> [Sequence]``, registered as
``register("skill", "<plugin>.<name>")`` and named by workflow steps. It decides
*what* the tool does (poses, gripper, what may be touched); the framework finds
the joints (``rsf.planning.teach``), plans the moves between them and times them
(``rsf.planning.plan``), and checks every sample against the same rules.

A :class:`~rsf.planning.teach.Waypoint` is

* ``T``: the tool pose (``robot.tool_frame``, here ``tcp``) in the world;
* ``scene``: what must not be hit there (world solids) and what the robot holds
  (``attached``: ``(name, link, solids in the object frame, T_link_object)``);
* ``policy``: the contact rules (a margin per robot link / obstacle pair, or
  contact allowed) -- ``rsf.collision.ContactPolicy``;
* ``joints``: the other joints (the fingers) while arriving and there;
* ``arrive``: ``free`` (any collision-free motion, planned later), ``line`` (a
  straight tool line, one continuous IK branch, every 5 mm checked) or ``path``;
* ``action``: done on arrival (a hold segment: the fingers move to the next
  waypoint's values); ``meta["caps"]``: the Cartesian speed limits of the arrival
  (a key of ``process.motion.caps``).

Conventions the rest of the framework reads (docs/new_cell.md):

* sequence names ``pick:<feature key>`` and ``place:<feature key>``, and
  ``meta["slot"]``: the planner tracks what is where, the judge finds transfers;
* a sequence that ends holding something has it in its last waypoint's
  ``scene.attached``; a place's last waypoint scene has it where it was put
  (the judge's placement reference).

Parameters, ``process.pick`` in the layout (defaults below)::

    pick: {approach: 0.10, open_clearance: 0.015, squeeze: 0.002, margin: 0.005}
"""
from __future__ import annotations

import math

import numpy as np

from rsf.collision import ContactPolicy, ContactRule
from rsf.core import se3
from rsf.model.registry import register
from rsf.planning.teach import Scene, Sequence, Waypoint

from .geometry import FINGER_LINKS, FINGERS, block_size, gripper

DEFAULTS = {"approach": 0.10, "open_clearance": 0.015, "squeeze": 0.002, "margin": 0.005}


def params(world) -> dict:
    return {**DEFAULTS, **(world.process.get("pick") or {})}


def grasp_pose(world, key: str) -> np.ndarray:
    """The tool at the block's centre on feature *key*, pointing down (tool z = -world z),
    the fingers closing along the feature's x."""
    f = world.feature(key)
    h = block_size(world, key)[2]
    return np.asarray(f.T) @ se3.make_T(se3.rot_x(math.pi), (0.0, 0.0, h / 2))


def above(T: np.ndarray, d: float) -> np.ndarray:
    out = np.array(T, float)
    out[2, 3] += d
    return out


def finger_values(world, key: str) -> tuple:
    """``(open, closed)`` finger joint values for the block at *key*. Closed is
    ``squeeze`` inside the block's half width: the fingers press on it (the
    simulator's finger force is what holds it)."""
    p, g = params(world), gripper(world)
    half = block_size(world, key)[0] / 2
    opened = min(g["finger_upper"], half + p["open_clearance"])
    closed = max(g["finger_lower"], half - p["squeeze"])
    return {j: opened for j in FINGERS}, {j: closed for j in FINGERS}


def blocks(world, exclude=()) -> tuple:
    """Every block template's solids except those at the *exclude* features: a
    neighbour may stand on every other slot (the worst case)."""
    return tuple(s for o in world.objects if o.at not in exclude for s in o.world_solids())


def rules(world, key: str, *, holding: bool) -> ContactPolicy:
    """Contact rules at slot *key*: everything keeps ``margin``, except

    * the fingers may touch the block at *key* (they grip it), the rest of the
      gripper keeps 0 from it;
    * a held block may stand on its slot's support (the feature attribute
      ``support``, margin 0: it starts and ends there);
    * the arm's base may stand on the floor (``robot_seat``)."""
    p = params(world)
    support = world.feature(key).get("support")
    if not support:
        raise ValueError(f"{key}: the slot has no 'support' attribute (the solid it rests on)")
    own = f"{world.object_at(key).name}/*"
    out = [ContactRule(allow=True, links=FINGER_LINKS, names=(own,), label="fingers grip the block"),
           ContactRule(margin=0.0, names=(own,), label="the block being handled"),
           ContactRule(allow=True, links=("link_0",), tags=("robot_seat",), label="base on the floor")]
    if holding:
        out.insert(0, ContactRule(margin=0.0, links=("attached/*",), names=(support,), label="held block on its support"))
    return ContactPolicy(p["margin"], out)


def _held(world, key: str):
    """``(name, link, solids, T_link_object)``: the block as held after a grasp at *key*."""
    obj = world.object_at(key)
    return ("block", world.robot.tool_frame, obj.solids, se3.inv(grasp_pose(world, key)) @ np.asarray(obj.T))


@register("skill", "example.pick")
def pick(world, args: dict) -> list:
    key = args.get("at", args.get("from"))
    T = grasp_pose(world, key)
    up = above(T, params(world)["approach"])
    opened, closed = finger_values(world, key)
    static = tuple(world.solids(objects=False))
    with_block = Scene(static + blocks(world))
    holding = Scene(static + blocks(world, exclude=(key,)), (_held(world, key),))
    free_rules, held_rules = rules(world, key, holding=False), rules(world, key, holding=True)
    wps = [
        Waypoint("above", up, with_block, free_rules, opened, "free"),
        Waypoint("grasp", T, with_block, free_rules, opened, "line", action="close the fingers",
                 meta={"caps": "near"}),
        Waypoint("lift", up, holding, held_rules, closed, "line", meta={"caps": "near"}),
    ]
    return [Sequence(f"pick:{key}", wps, {"skill": "example.pick", "slot": key})]


@register("skill", "example.place")
def place(world, args: dict) -> list:
    key = args.get("at", args.get("to"))
    T = grasp_pose(world, key)
    up = above(T, params(world)["approach"])
    opened, closed = finger_values(world, key)
    static = tuple(world.solids(objects=False))
    holding = Scene(static + blocks(world, exclude=(key,)), (_held(world, key),))
    placed = Scene(static + blocks(world))
    held_rules, free_rules = rules(world, key, holding=True), rules(world, key, holding=False)
    wps = [
        Waypoint("above", up, holding, held_rules, closed, "free"),
        Waypoint("place_down", T, holding, held_rules, closed, "line", action="open the fingers",
                 meta={"caps": "near"}),
        Waypoint("retreat", up, placed, free_rules, opened, "line", meta={"caps": "near"}),
    ]
    return [Sequence(f"place:{key}", wps, {"skill": "example.place", "slot": key})]


@register("skill", "example.transfer")
def transfer(world, args: dict) -> list:
    """Pick at ``from``, place at ``to``; the move between them is a free move (planned later)."""
    return pick(world, {"at": args["from"]}) + place(world, {"at": args["to"]})

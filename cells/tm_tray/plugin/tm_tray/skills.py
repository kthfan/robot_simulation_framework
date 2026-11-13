"""TM tray skills: pick, place and transfer as teach-engine sequences (P3).

Each :class:`sequence.Phase` becomes a :class:`rsf.planning.teach.Waypoint` with

* the scene at that phase: every other tray in its slot (the worst case for a
  neighbour, as in M3), the target tray seated where the phase has it or
  attached to the flange, registered against the push plate;
* the contact rules of :mod:`collision_check` (target jig and tray margin 0, the
  jaws may touch the target tray, contact without penetration while pushing);
* the jaws (D5, D30): open on the way in, closed to the tray width after
  ``lift``; on place, opened above the columns before the tray goes down;
* how it is reached: straight lines inside the shelf, the tilt motion
  (``process.tilt_motion``, with its own clearance), free moves elsewhere.

While the plate lifts the tray off its columns (``push -> lift``) or sets it
down (``place_lift -> place_down``) the tray is in transit between the jig and
the plate: along those lines the robot may touch the target tray; the carried
tray's own clearance is judged at the ends.
"""
from __future__ import annotations

from dataclasses import replace

from rsf.collision import ContactPolicy, ContactRule
from rsf.core import se3
from rsf.model.registry import register
from rsf.planning.teach import Scene, Sequence, Waypoint

from . import hardware as A
from . import sequence as SEQ
from .collision_check import JAWS, MARGIN, carried_tray_pose, path_policy, policy
from .geometry import slot_of
from .tool import tool_geometry

PICK_ARRIVE = {"sight_far": "free", "sight_near": "line", "pre_insert": "free", "insert": "line",
               "push": "line", "lift": "line", "withdraw": "line", "carry": "path"}
PLACE_ARRIVE = {"carry": "free", "withdraw": "path", "place_lift": "line", "place_down": "line",
                "retreat": "line"}
PICK_JAW = {"withdraw": A.JAW_Q_FIT, "carry": A.JAW_Q_FIT}                 # else open
PLACE_JAW = {"carry": A.JAW_Q_FIT, "withdraw": A.JAW_Q_FIT, "place_lift": A.JAW_Q_FIT}
PICK_ACTION = {"sight_far": "read the AprilGrid (far)", "sight_near": "read the AprilGrid (near)",
               "push": "tray pushed against the push plate", "lift": f"close the jaws to {A.JAW_Q_FIT:.4f}",
               "carry": "tilted for transport"}
PLACE_ACTION = {"place_lift": f"open the jaws to {A.JAW_Q_OPEN:.4f}", "place_down": "tray set down"}


def transit_policy(key: str) -> ContactPolicy:
    """While the tray passes between its jig and the plate."""
    return ContactPolicy(MARGIN, [
        ContactRule(allow=True, names=(f"tray/{key}/*",), label="tray in transit jig <-> plate"),
        ContactRule(margin=0.0, names=(f"jig/{key}/*",), label="target jig"),
        ContactRule(allow=True, links=("link_0",), tags=("robot_seat",), label="base on its mount"),
    ])


def phase_scene(world, key: str, ph) -> Scene:
    """The scene of a :class:`sequence.Phase` at slot *key*."""
    tray = world.object_at(key)
    if tray is None:
        return Scene(tuple(world.solids()))
    others = tuple(world.solids(exclude_objects=[tray.name]))
    if ph.tray == "carried":
        return Scene(others, (("tray", "flange", tray.solids, carried_tray_pose(ph.carried_z)),))
    moved = replace(tray, T=tray.T @ se3.make_T(None, (0.0, ph.shift, 0.0)))   # jig +y: toward the robot
    return Scene(others + tuple(moved.world_solids()))


def _waypoints(world, key, phases, arrive, jaw, action) -> list:
    out, scenes = [], {}
    for i, ph in enumerate(phases):
        scene = phase_scene(world, key, ph)
        scenes[ph.name] = scene
        along_scene = along_policy = None
        if ph.name in ("lift", "place_down"):                 # tray between jig and plate
            along_scene = scenes[phases[i - 1].name] if ph.name == "lift" else scene
            along_policy = transit_policy(key)
        if ph.path_clearance is not None:
            along_policy = path_policy(ph.path_clearance)
        j = jaw.get(ph.name, A.JAW_Q_OPEN)
        out.append(Waypoint(ph.name, ph.T, scene, policy(key, contact=ph.contact),
                            {n: j for n in JAWS}, arrive[ph.name] if i else "free", ph.path,
                            along_policy, along_scene, action.get(ph.name, ""),
                            {"tray": ph.tray, "tray_shift": ph.shift}))
    return out


@register("skill", "tm_tray.pick")
def pick(world, args: dict) -> list:
    key = args["at"] if "at" in args else args["from"]
    phases = SEQ.pick_phases(world.process, slot_of(world.feature(key)), tool_geometry(world))
    return [Sequence(f"pick:{key}", _waypoints(world, key, phases, PICK_ARRIVE, PICK_JAW, PICK_ACTION),
                     {"skill": "tm_tray.pick", "slot": key})]


@register("skill", "tm_tray.place")
def place(world, args: dict) -> list:
    key = args["at"] if "at" in args else args["to"]
    phases = SEQ.place_phases(world.process, slot_of(world.feature(key)), tool_geometry(world))
    return [Sequence(f"place:{key}", _waypoints(world, key, phases, PLACE_ARRIVE, PLACE_JAW, PLACE_ACTION),
                     {"skill": "tm_tray.place", "slot": key})]


@register("skill", "tm_tray.transfer")
def transfer(world, args: dict) -> list:
    """Pick at ``from``, place at ``to`` (the move between them is free, P4)."""
    return pick(world, {"at": args["from"]}) + place(world, {"at": args["to"]})

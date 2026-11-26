"""Schedulers decide which features a workflow cycle works on next.

A scheduler is registered as ``register("scheduler", name)`` and constructed as
``cls(world, sets, **params)``. Each call to :meth:`next` gets the current
occupancy and returns either a bindings dict (``{"source": key, ...}``) or an
:class:`Event` that ends the run (a refill or removal request, D1).

``rsf.ordered`` covers the common case: every binding takes the first feature,
in a fixed order, whose state matches::

    scheduler:
      type: rsf.ordered
      params:
        bind:
          source:      {set: incoming, order: [-height, column], state: occupied, on_none: request_refill}
          destination: {set: outgoing, order: [height, column],  state: empty,    on_none: request_removal}

Binding keys: ``set`` (a workflow set), ``order`` (feature attributes or ``x``/``y``/``z``;
a leading ``-`` sorts descending), ``state`` (``occupied`` / ``empty`` / ``any``, default
``any``), ``on_none`` (the end event's name, default ``<binding>_exhausted``). It keeps no
memory between cycles: the run ends only when the cycle's effects change the occupancy so
that a binding finds nothing (a cycle that puts things back where they were repeats forever).

``rsf.each`` visits every feature of a set once, in order, whatever happens to it
(round trips, inspections)::

    scheduler:
      type: rsf.each
      params:
        bind: {slot: {set: source, order: [index], state: occupied}}   # exactly one binding
        on_done: all_visited

A feature not in the wanted ``state`` when its turn comes is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from rsf.model.registry import register
from rsf.model.select import sort_features

STATES = ("occupied", "empty", "any")


@dataclass(frozen=True)
class Event:
    """Why a run stopped, and which binding found nothing."""
    name: str
    binding: str
    detail: str = ""


@runtime_checkable
class Scheduler(Protocol):
    def next(self, occupancy: dict):
        """Bindings ``{name: feature_key}`` for the next cycle, or an :class:`Event`."""


@register("scheduler", "rsf.ordered")
class OrderedScheduler:
    def __init__(self, world, sets: dict, bind: dict):
        if not bind:
            raise ValueError("rsf.ordered: no bindings")
        self.bind = {}
        for name, spec in bind.items():
            unknown = set(spec) - {"set", "order", "state", "on_none"}
            if unknown:
                raise ValueError(f"binding {name!r}: unknown key(s) {sorted(unknown)}")
            if spec["set"] not in sets:
                raise ValueError(f"binding {name!r}: no set {spec['set']!r}")
            state = spec.get("state", "any")
            if state not in STATES:
                raise ValueError(f"binding {name!r}: state {state!r} not in {STATES}")
            order = sort_features(sets[spec["set"]], spec.get("order", []))
            self.bind[name] = (order, state, spec.get("on_none", f"{name}_exhausted"))

    def order(self, name: str) -> list:
        """The fixed visiting order of binding *name* (feature keys)."""
        return [f.key for f in self.bind[name][0]]

    def next(self, occupancy: dict):
        taken, out = set(), {}
        for name, (order, state, on_none) in self.bind.items():
            pick = next((f for f in order if f.key not in taken and _state_ok(occupancy, f.key, state)), None)
            if pick is None:
                return Event(on_none, name, f"no {state} feature left for {name!r}")
            taken.add(pick.key)
            out[name] = pick.key
        return out


def _state_ok(occupancy: dict, key: str, state: str) -> bool:
    return state == "any" or (state == "occupied") == (key in occupancy)


@register("scheduler", "rsf.each")
class EachScheduler:
    """Every feature of one set once, in a fixed order (module docstring)."""

    def __init__(self, world, sets: dict, bind: dict, on_done: str = "all_visited"):
        if not bind or len(bind) != 1:
            raise ValueError(f"rsf.each: needs exactly one binding, got {sorted(bind or {})}")
        ((name, spec),) = bind.items()
        unknown = set(spec) - {"set", "order", "state"}
        if unknown:
            raise ValueError(f"binding {name!r}: unknown key(s) {sorted(unknown)}")
        if spec.get("set") not in sets:
            raise ValueError(f"binding {name!r}: no set {spec.get('set')!r}")
        self.state = spec.get("state", "any")
        if self.state not in STATES:
            raise ValueError(f"binding {name!r}: state {self.state!r} not in {STATES}")
        self.name = name
        self.queue = [f.key for f in sort_features(sets[spec["set"]], spec.get("order", []))]
        self._order = list(self.queue)
        self.on_done = str(on_done)

    def order(self, name: str) -> list:
        return list(self._order)

    def next(self, occupancy: dict):
        while self.queue:
            key = self.queue.pop(0)
            if _state_ok(occupancy, key, self.state):
                return {self.name: key}
        return Event(self.on_done, self.name, f"every feature of {self.name!r} visited")

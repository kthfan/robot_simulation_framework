"""Which slot to service next.

D1 fixes the default policy — drain the incoming rack top-to-bottom, then
left-to-right — but explicitly keeps the scheduler replaceable. Anything
satisfying :class:`Scheduler` can be substituted; the protocol is
``runtime_checkable`` so a host can assert its plug-in fits before starting.

Working top-down is not arbitrary: removing a tray from an upper shelf cannot
disturb the ones below it, so the world model stays valid for longer.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Scheduler(Protocol):
    """Chooses the order in which slots are serviced."""

    def order(self) -> list:
        """Remaining source slots, in the order they will be served."""

    def next_slot(self):
        """Pop the next source slot, or ``None`` when the run is finished."""

    def destinations(self) -> list:
        """Candidate slots to place trays into, best first."""


class SequentialScheduler:
    """The default policy: drain the incoming rack top-down, left-to-right.

    *occupied*, when given, restricts the run to the slots believed to hold a
    tray. Passing ``None`` means "assume every slot is full", which is only
    correct before the scanning stage has run.
    """

    def __init__(self, slots, occupied=None,
                 source_role: str = "incoming", dest_role: str = "outgoing"):
        sources = [s for s in slots if s.role == source_role]
        if occupied is not None:
            sources = [s for s in sources if s.key in occupied]
        self._queue = sorted(sources, key=lambda s: (-s.height, s.column))
        # Fill the outgoing rack from the bottom up, so a partially loaded rack
        # keeps its centre of mass low and never has a tray above an empty slot.
        self._dests = sorted((s for s in slots if s.role == dest_role),
                             key=lambda s: (s.height, s.column))

    def order(self) -> list:
        return list(self._queue)

    def next_slot(self):
        return self._queue.pop(0) if self._queue else None

    def destinations(self) -> list:
        return list(self._dests)

    def __len__(self) -> int:
        return len(self._queue)

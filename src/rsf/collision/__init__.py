"""Collision checking against a :class:`~rsf.model.world.World`.

The vocabulary is backend-independent:

* **Contact rules** decide, per (robot link, obstacle) pair, the clearance that
  pair must keep. The first matching rule wins; a rule with ``allow=True``
  removes the pair (it may touch). This is MoveIt's allowed-collision matrix
  plus per-pair margins, which cuRobo does not have yet (docs/PLAN.md §6.1).
* A **report** gives, per configuration, the worst violation and the nearest
  obstacle per group, so a failure names what was hit and by how much.

Backends: :mod:`rsf.collision.coal_backend` (CPU, exact triangle meshes).
cuRobo (GPU) is used for cross-checks and will implement the same interface.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class ContactRule:
    """Applies when the robot link matches *links* and the obstacle has every
    tag in *tags* (and its name matches *names*)."""
    margin: float = 0.0
    links: tuple = ("*",)
    tags: tuple = ()
    names: tuple = ("*",)
    allow: bool = False
    label: str = ""

    def matches(self, link: str, solid) -> bool:
        return (any(fnmatch.fnmatchcase(link, p) for p in self.links)
                and set(self.tags) <= set(solid.tags)
                and any(fnmatch.fnmatchcase(solid.name, p) for p in self.names))


@dataclass
class ContactPolicy:
    default_margin: float
    rules: list = field(default_factory=list)

    def resolve(self, link: str, solid):
        """``(margin, label)`` for the pair, or ``None`` if contact is allowed."""
        for r in self.rules:
            if r.matches(link, solid):
                return None if r.allow else (r.margin, r.label or "rule")
        return self.default_margin, "default"


@dataclass(frozen=True)
class PairResult:
    link: str
    obstacle: str
    distance: float          #: signed: negative is penetration (when the backend knows it)
    margin: float
    rule: str

    @property
    def slack(self) -> float:
        return self.distance - self.margin


@dataclass
class ConfigReport:
    ok: bool
    worst: Optional[PairResult]                      #: smallest slack among checked pairs
    violations: list = field(default_factory=list)  #: pairs with slack < 0
    self_violations: list = field(default_factory=list)
    nearest_by_group: dict = field(default_factory=dict)  #: group -> PairResult

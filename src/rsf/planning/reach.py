"""Reachability: which target poses have an exact, in-limit IK solution.

Targets come from a registered ``targets`` provider,
``fn(world, feature) -> {phase: T_world}``, so the same sweep serves any cell:
the TM tray plugin registers ``tm_tray.grasp`` (the seven grasp/release poses
of a slot). A closed-form solver returning nothing is a statement about the
workspace (AI-Robotics-V2 D21), so this is the authority on reach; collisions
and paths are judged later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rsf.model import registry


@dataclass
class TargetReach:
    feature: str
    phases: dict = field(default_factory=dict)   #: phase -> list of joint vectors

    @property
    def missing(self) -> list:
        return [p for p, qs in self.phases.items() if not qs]

    @property
    def ok(self) -> bool:
        return not self.missing


def sweep(world, arm, provider: str, select=None) -> list:
    """Reach of every selected feature's targets (default: all features the provider accepts)."""
    fn = registry.get("targets", provider)
    out = []
    for f in world.features(select):
        targets = fn(world, f)
        if not targets:
            continue
        r = TargetReach(f.key)
        for phase, T in targets.items():
            r.phases[phase] = arm.ik(T, expand_turns=False)
        out.append(r)
    return out


def summary(results: list) -> dict:
    return {"targets": len(results), "reachable": sum(r.ok for r in results),
            "poses": sum(len(r.phases) for r in results),
            "poses_reachable": sum(bool(q) for r in results for q in r.phases.values()),
            "unreachable": {r.feature: r.missing for r in results if not r.ok}}

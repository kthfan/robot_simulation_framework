"""Audit a world: run every applicable check, never raise, return a diffable report.

Rules (from AI-Robotics-V2 ``cell_core.audit``):

* **An audit never raises.** A crashing check becomes a ``skip`` with the reason,
  so last week's report and this week's can always be diffed.
* **A skipped check is not a passing check.** :func:`skipped` lists them.

Checks are registered as ``register("check", "<plugin>.<name>")`` with the
signature ``fn(world) -> list[Check]``. A layout may restrict which run with a
top-level ``audit: [glob, ...]``; by default every registered check runs, and a
check that does not apply to the world returns ``[]``.
"""
from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass
from typing import Optional

from rsf.model import registry

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"
STATUSES = (FAIL, WARN, SKIP, PASS)  #: worst first


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    value: Optional[float] = None

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def check(name, ok, detail_ok, detail_bad, value=None, soft=False) -> Check:
    """A check from a boolean; *soft* turns a failure into a warning."""
    if ok:
        return Check(name, PASS, detail_ok, value)
    return Check(name, WARN if soft else FAIL, detail_bad, value)


def audit(world, only=None) -> list:
    patterns = only if only is not None else (world.source.get("audit") or ["*"])
    out = []
    for name in registry.names("check"):
        if not any(fnmatch.fnmatchcase(name, p) for p in patterns):
            continue
        try:
            out += list(registry.get("check", name)(world))
        except Exception as exc:  # noqa: BLE001 - an audit never raises
            out.append(Check(name, SKIP, f"could not be evaluated: {type(exc).__name__}: {exc}"))
    return out


def verdict(checks) -> str:
    return FAIL if any(c.failed for c in checks) else PASS


def failures(checks) -> list:
    return [c for c in checks if c.status == FAIL]


def warnings(checks) -> list:
    return [c for c in checks if c.status == WARN]


def skipped(checks) -> list:
    return [c.name for c in checks if c.status == SKIP]


def to_dict(checks) -> list:
    return [asdict(c) for c in checks]


# ── framework-level checks ──────────────────────────────────────────────────
@registry.register("check", "rsf.unique_solid_names")
def _unique_solid_names(world) -> list:
    names = [s.name for s in world.solids()]
    dup = sorted({n for n in names if names.count(n) > 1})
    return [check("rsf.unique_solid_names", not dup, f"{len(names)} solids, names unique",
                  f"duplicate solid names: {dup[:10]}")]


@registry.register("check", "rsf.robot_model")
def _robot_model(world) -> list:
    if world.robot is None:
        return [Check("rsf.robot_model", WARN, "the layout has no robot")]
    from rsf.model.urdf import UrdfModel
    m = UrdfModel.from_file(world.robot.urdf)
    missing = [g.filename for l in m.links.values() for g in l.collisions if g.kind == "mesh" and g.path is None]
    out = [check("rsf.robot_model/meshes", not missing, "every robot collision mesh resolves",
                 f"missing robot meshes: {missing[:5]}")]
    out.append(check("rsf.robot_model/tool_frame", world.robot.tool_frame in m.links,
                     f"tool frame {world.robot.tool_frame!r} is a link",
                     f"tool frame {world.robot.tool_frame!r} is not a link of {m.name}"))
    return out

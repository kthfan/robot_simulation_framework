"""Workflows: what the cell does, as YAML, expanded into concrete task instances.

::

    schema: rsf.workflow/1
    name: inspect_and_sort
    plugins: []                       # optional, like layouts
    sets:                             # named feature selections (rsf.model.select)
      incoming:   {kind: slot, role: incoming}
      outgoing:   {kind: slot, role: outgoing}
      inspection: {kind: slot, role: inspection}
    initial:                          # occupancy to plan from (else the layout's)
      tray: [incoming]                #   object -> set names or feature keys
    scheduler:
      type: rsf.ordered
      params: {bind: {source: {...}, destination: {...}}}
    cycle:                            # repeated for every scheduler binding
      - {skill: transfer, args: {object: tray, from: $source, to: $inspection}}
      - {skill: wait, args: {signal: inspection_done}}
      - {skill: transfer, args: {object: tray, from: $inspection, to: $destination}}
    params: {...}                     # passed to every skill (P3)

``$name`` in an argument is a scheduler binding, or a set that holds exactly one
feature. A step's state change (its *effect*) is looked up by ``effect:`` or,
by default, by the skill's name in the ``effect`` registry; skills with no
registered effect do not change occupancy. Skills themselves (motion) are P3:
expansion only decides *what* happens *where*, in order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from rsf.model import registry
from rsf.orchestrator import scheduler as _schedulers  # noqa: F401  (registers rsf.ordered)
from rsf.orchestrator.scheduler import Event

SCHEMA = "rsf.workflow/1"
TOP_KEYS = {"schema", "name", "description", "plugins", "sets", "initial", "scheduler", "cycle", "params",
            "max_cycles"}


class WorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    skill: str
    args: dict
    effect: Optional[str] = None


@dataclass
class Workflow:
    name: str
    sets: dict
    scheduler: dict
    cycle: list
    initial: Optional[dict] = None
    params: dict = field(default_factory=dict)
    max_cycles: int = 1000
    description: str = ""

    @classmethod
    def from_file(cls, path) -> "Workflow":
        path = Path(path).resolve()
        doc = yaml.safe_load(path.read_text()) or {}
        unknown = set(doc) - TOP_KEYS
        if unknown:
            raise WorkflowError(f"{path.name}: unknown key(s) {sorted(unknown)}")
        if doc.get("schema", SCHEMA) != SCHEMA:
            raise WorkflowError(f"{path.name}: schema {doc.get('schema')!r}, expected {SCHEMA}")
        for p in doc.get("plugins", []):
            registry.load_plugin(path.parent / p)
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc: dict) -> "Workflow":
        cycle = []
        for i, s in enumerate(doc.get("cycle") or []):
            unknown = set(s) - {"skill", "args", "effect"}
            if unknown or "skill" not in s:
                raise WorkflowError(f"cycle step {i}: needs 'skill'; unknown key(s) {sorted(unknown)}")
            cycle.append(Step(s["skill"], dict(s.get("args") or {}), s.get("effect")))
        if not cycle:
            raise WorkflowError("empty cycle")
        return cls(doc.get("name", ""), dict(doc.get("sets") or {}), dict(doc["scheduler"]), cycle,
                   doc.get("initial"), dict(doc.get("params") or {}), int(doc.get("max_cycles", 1000)),
                   doc.get("description", ""))

    # ── resolution ─────────────────────────────────────────────────────────
    def resolve_sets(self, world) -> dict:
        out = {}
        for name, sel in self.sets.items():
            fs = world.features(sel)
            if not fs:
                raise WorkflowError(f"set {name!r}: selector {sel!r} matched no feature")
            out[name] = fs
        return out

    def initial_occupancy(self, world, sets: dict) -> dict:
        """``{feature_key: object_name}`` to plan from."""
        if self.initial is None:
            return dict(world.occupancy())
        occ = {}
        for obj, places in self.initial.items():
            for p in places:
                keys = [f.key for f in sets[p]] if p in sets else [p]
                for k in keys:
                    world.feature(k)  # must exist
                    occ[k] = f"{obj}/{k}"
        return occ


@dataclass(frozen=True)
class TaskInstance:
    index: int
    cycle: int
    skill: str
    args: dict
    bindings: dict


@dataclass
class Expansion:
    tasks: list
    end: Event
    occupancy: dict          #: after the last task
    sets: dict

    @property
    def cycles(self) -> int:
        return 1 + max((t.cycle for t in self.tasks), default=-1)


def expand(workflow: Workflow, world, occupancy: Optional[dict] = None) -> Expansion:
    """The concrete task list the workflow produces on *world*, cycle by cycle,
    until the scheduler raises an event or ``max_cycles`` is reached."""
    sets = workflow.resolve_sets(world)
    occ = dict(occupancy if occupancy is not None else workflow.initial_occupancy(world, sets))
    sch = registry.get("scheduler", workflow.scheduler["type"])(world, sets, **(workflow.scheduler.get("params") or {}))
    tasks, end = [], None
    for c in range(workflow.max_cycles):
        b = sch.next(occ)
        if isinstance(b, Event):
            end = b
            break
        for step in workflow.cycle:
            args = {k: _resolve(v, b, sets) for k, v in step.args.items()}
            eff = step.effect or step.skill
            if eff != "none" and eff in registry.names("effect"):
                registry.get("effect", eff)(occ, args)
            tasks.append(TaskInstance(len(tasks), c, step.skill, args, dict(b)))
    if end is None:
        end = Event("max_cycles", "", f"stopped after {workflow.max_cycles} cycles")
    return Expansion(tasks, end, occ, sets)


def _resolve(v, bindings: dict, sets: dict):
    if isinstance(v, str) and v.startswith("$"):
        name = v[1:]
        if name in bindings:
            return bindings[name]
        if name in sets:
            if len(sets[name]) != 1:
                raise WorkflowError(f"${name}: set has {len(sets[name])} features, needs exactly one")
            return sets[name][0].key
        raise WorkflowError(f"${name}: neither a binding nor a set")
    if isinstance(v, list):
        return [_resolve(x, bindings, sets) for x in v]
    if isinstance(v, dict):
        return {k: _resolve(x, bindings, sets) for k, x in v.items()}
    return v


# ── built-in effects ────────────────────────────────────────────────────────
@registry.register("effect", "transfer")
def _transfer(occ: dict, args: dict) -> None:
    """Move the object at ``from`` to ``to``."""
    src, dst = args["from"], args["to"]
    if src not in occ:
        raise WorkflowError(f"transfer: nothing at {src}")
    if dst in occ:
        raise WorkflowError(f"transfer: {dst} is occupied by {occ[dst]}")
    occ[dst] = occ.pop(src)  # the object keeps its identity

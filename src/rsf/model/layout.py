"""Layouts: a cell described in YAML, compiled into a :class:`~rsf.model.world.World`.

::

    schema: rsf.layout/1
    name: strict
    extends: as_built.yaml          # deep-merged underneath this file (optional)
    plugins: [../plugin/tm_tray]    # imported before anything is built
    process: {...}                  # free-form cell process parameters
    entities:                       # built in order; later ones may use earlier features
      pedestal: {generator: tm_tray.pedestal, params: {...}}
      fence:    {urdf: assets/fence.urdf, pose: {xyz: [0, 1, 0], rpy: [0, 0, 0]}}
    placements:                     # one fixed copy per selected feature
      jig: {urdf: ..., at: {kind: slot}, offset: {xyz: [...]}}
    objects:                        # movable things, initially at selected features
      tray: {urdf: ..., at: {kind: slot}, offset: {...}, initially: all}
    robot: {urdf: ..., base: {feature: pedestal/arm_base}, tool_frame: flange}

Paths are relative to the YAML file that wrote them. Overrides are dotted paths
into the resolved document (``entities.pedestal.params.leg_height``); a path
that does not exist is an error, so a typo cannot silently do nothing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from rsf.core import se3
from rsf.model import generators as _builtin_generators  # noqa: F401  (registers rsf.*)
from rsf.model import registry
from rsf.model.urdf import UrdfModel
from rsf.model.world import Entity, Feature, ObjectInstance, RobotSpec, Solid, World, solids_from_urdf

SCHEMA = "rsf.layout/1"
TOP_KEYS = {"schema", "name", "description", "extends", "plugins", "process", "entities",
            "placements", "objects", "robot", "audit"}
_PATH_KEYS = {"urdf", "mesh"}


class LayoutError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Loading: extends, paths, overrides
# ─────────────────────────────────────────────────────────────────────────────
def deep_merge(base: dict, top: dict) -> dict:
    """*top* over *base*; mappings merge, everything else (lists too) replaces."""
    out = copy.deepcopy(base)
    for k, v in top.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _absolutise(doc: dict, base_dir: Path) -> dict:
    """Make file references absolute so merged documents keep their meaning."""
    def walk(node, key=None):
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            if key == "plugins":
                return [str((base_dir / p).resolve()) for p in node]
            return [walk(v) for v in node]
        if key in _PATH_KEYS and isinstance(node, str):
            return str((base_dir / node).resolve())
        return node
    return walk(doc)


def load_document(path, _seen=None) -> dict:
    """The layout at *path* with ``extends`` resolved and paths made absolute."""
    path = Path(path).resolve()
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise LayoutError(f"extends cycle at {path}")
    _seen.add(path)
    doc = yaml.safe_load(path.read_text()) or {}
    unknown = set(doc) - TOP_KEYS
    if unknown:
        raise LayoutError(f"{path.name}: unknown top-level key(s) {sorted(unknown)}")
    parent = doc.pop("extends", None)
    doc = _absolutise(doc, path.parent)
    if parent is not None:
        base = load_document(path.parent / parent, _seen)
        if "plugins" in doc:  # plugins accumulate
            doc["plugins"] = list(dict.fromkeys(base.get("plugins", []) + doc["plugins"]))
        doc = deep_merge(base, doc)
    if doc.get("schema", SCHEMA) != SCHEMA:
        raise LayoutError(f"{path.name}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    doc.setdefault("schema", SCHEMA)
    return doc


def get_path(doc: dict, dotted: str):
    node = doc
    for part in dotted.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise LayoutError(f"no {dotted!r} in the layout (stopped at {part!r})")
    return node


def set_path(doc: dict, dotted: str, value) -> dict:
    """A copy of *doc* with the existing entry at *dotted* replaced."""
    get_path(doc, dotted)  # must exist
    out = copy.deepcopy(doc)
    parts = dotted.split(".")
    node = out
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return out


def apply_overrides(doc: dict, overrides: Optional[dict]) -> dict:
    for k, v in (overrides or {}).items():
        doc = set_path(doc, k, v)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Building
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BuildContext:
    """What a generator may look at: the document and the entities built so far."""
    doc: dict
    entities: list = field(default_factory=list)

    @property
    def process(self) -> dict:
        return self.doc.get("process", {})

    def feature(self, key: str) -> Feature:
        for e in self.entities:
            for f in e.features:
                if f.key == key:
                    return f
        raise LayoutError(f"no feature {key!r} built yet")

    def features(self, select=None) -> list:
        from rsf.model.select import select_features
        fs = [f for e in self.entities for f in e.features]
        return fs if select is None else select_features(fs, select)


def parse_pose(spec: Optional[dict], ctx: Optional[BuildContext] = None) -> np.ndarray:
    """``{xyz, rpy}``, ``{xyz, quat_wxyz}``, ``{matrix}`` or ``{feature, offset}``."""
    if spec is None:
        return np.eye(4)
    spec = dict(spec)
    base = np.eye(4)
    if "feature" in spec:
        if ctx is None:
            raise LayoutError("a feature-relative pose needs a build context")
        base = ctx.feature(spec.pop("feature")).T
        spec = spec.pop("offset", {}) or {}
    if "matrix" in spec:
        return base @ np.asarray(spec["matrix"], dtype=float).reshape(4, 4)
    unknown = set(spec) - {"xyz", "rpy", "quat_wxyz"}
    if unknown:
        raise LayoutError(f"pose keys {sorted(unknown)}")
    R = (se3.quat_to_matrix(spec["quat_wxyz"]) if "quat_wxyz" in spec
         else se3.rpy_to_matrix(spec.get("rpy", (0.0, 0.0, 0.0))))
    return base @ se3.make_T(R, np.asarray(spec.get("xyz", (0.0, 0.0, 0.0)), dtype=float))


def _urdf_entity(name: str, spec: dict, ctx: BuildContext) -> Entity:
    model = UrdfModel.from_file(spec["urdf"])
    T = parse_pose(spec.get("pose"), ctx)
    tags = frozenset(spec.get("tags", [])) | {name}
    solids = tuple(s.moved(T) for s in solids_from_urdf(model, prefix=f"{name}/", entity=name, tags=tags))
    return Entity(name, spec.get("kind", "static"), solids, (), {"urdf": spec["urdf"]})


def _build_entity(name: str, spec: dict, ctx: BuildContext) -> Entity:
    if "generator" in spec:
        unknown = set(spec) - {"generator", "params"}
        if unknown:
            raise LayoutError(f"entity {name!r}: unknown key(s) {sorted(unknown)}")
        ent = registry.get("generator", spec["generator"])(name, dict(spec.get("params") or {}), ctx)
        if ent.name != name:
            raise LayoutError(f"generator {spec['generator']} returned entity {ent.name!r} for {name!r}")
        return ent
    if "urdf" in spec:
        return _urdf_entity(name, spec, ctx)
    raise LayoutError(f"entity {name!r} needs a generator or a urdf")


def _placements(doc: dict, ctx: BuildContext) -> list:
    out = []
    for pname, spec in (doc.get("placements") or {}).items():
        model = UrdfModel.from_file(spec["urdf"])
        local = solids_from_urdf(model)
        off = parse_pose(spec.get("offset"))
        targets = ctx.features(spec["at"])
        if not targets:
            raise LayoutError(f"placement {pname!r}: selector {spec['at']!r} matched no feature")
        for f in targets:
            ename = f"{pname}/{f.key}"
            tags = frozenset(spec.get("tags", [])) | {pname, f"at:{f.key}"}
            T = f.T @ off
            solids = tuple(Solid(f"{ename}/{s.name}", T @ s.T, s.shape, ename, tags) for s in local)
            out.append(Entity(ename, spec.get("kind", "fixture"), solids, (), {"at": f.key, "urdf": spec["urdf"]}))
    return out


def _objects(doc: dict, ctx: BuildContext, occupancy: Optional[dict]) -> list:
    out = []
    for oname, spec in (doc.get("objects") or {}).items():
        model = UrdfModel.from_file(spec["urdf"])
        local = tuple(Solid(s.name, s.T, s.shape, oname, frozenset({oname, "object"}))
                      for s in solids_from_urdf(model))
        mass = sum(l.mass or 0.0 for l in model.links.values()) or None
        off = parse_pose(spec.get("offset"))
        candidates = ctx.features(spec["at"])
        if occupancy is not None and oname in occupancy:
            keys = set(occupancy[oname])
        else:
            init = spec.get("initially", "all")
            keys = ({f.key for f in candidates} if init == "all" else set() if init == "none" else set(init))
        unknown = keys - {f.key for f in candidates}
        if unknown:
            raise LayoutError(f"object {oname!r}: not a place for it: {sorted(unknown)}")
        for f in candidates:
            if f.key in keys:
                out.append(ObjectInstance(f"{oname}/{f.key}", oname, local, f.T @ off, f.key, mass,
                                          Path(spec["urdf"])))
    return out


def build_world(doc: dict, occupancy: Optional[dict] = None) -> World:
    """Compile a resolved layout document.

    *occupancy* overrides an object's ``initially``: ``{"tray": ["incoming/L0/C0", ...]}``.
    """
    for p in doc.get("plugins", []):
        registry.load_plugin(p)
    ctx = BuildContext(doc)
    for name, spec in (doc.get("entities") or {}).items():
        ctx.entities.append(_build_entity(name, spec or {}, ctx))
    entities = list(ctx.entities) + _placements(doc, ctx)
    objects = _objects(doc, ctx, occupancy)
    robot = None
    if doc.get("robot"):
        r = doc["robot"]
        robot = RobotSpec(Path(r["urdf"]), parse_pose(r.get("base"), ctx), r.get("tool_frame", "tool0"),
                          {k: v for k, v in r.items() if k not in ("urdf", "base", "tool_frame")})
    names = [e.name for e in entities]
    if len(set(names)) != len(names):
        raise LayoutError(f"duplicate entity names: {sorted({n for n in names if names.count(n) > 1})}")
    return World(doc.get("name", ""), tuple(entities), tuple(objects), robot,
                 dict(doc.get("process") or {}), doc)


def load_world(path, overrides: Optional[dict] = None, occupancy: Optional[dict] = None) -> World:
    return build_world(apply_overrides(load_document(path), overrides), occupancy)

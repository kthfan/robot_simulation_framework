"""Named extension points and plugin loading.

A cell extends the framework by registering functions under a *kind* (for
example ``generator``) and a dotted name (``tm_tray.rack``). Layouts and
workflows refer to them by name, so a new rack type, scheduler or check is a
plugin, not a framework change.

Kinds used by the framework:

=============  =============================================================
``generator``  ``fn(name, params, ctx) -> Entity``: builds solids and features
``scheduler``  ``cls(world, spec, **params)``: orders task sources/targets
``check``      ``fn(world) -> list[Check]``: audit rules
``skill``      later (P3): expands a task into keyframes
=============  =============================================================
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

_REGISTRY: dict[str, dict[str, Callable]] = {}
_LOADED: dict[Path, object] = {}


class RegistryError(KeyError):
    pass


def register(kind: str, name: str, *, replace: bool = False):
    """Decorator: register *fn* as ``kind/name``."""
    def deco(fn):
        table = _REGISTRY.setdefault(kind, {})
        if name in table and table[name] is not fn and not replace:
            raise RegistryError(f"{kind} {name!r} is already registered")
        table[name] = fn
        return fn
    return deco


def get(kind: str, name: str) -> Callable:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "none"
        raise RegistryError(f"unknown {kind} {name!r} (registered: {known})") from None


def names(kind: str) -> list:
    return sorted(_REGISTRY.get(kind, {}))


def load_plugin(path) -> object:
    """Import a plugin package or module from a file path, once.

    *path* is a directory with ``__init__.py`` or a ``.py`` file. It is imported
    under ``rsf_plugins.<stem>`` so plugins never shadow installed packages, and
    relative imports inside a plugin package work.
    """
    path = Path(path).resolve()
    if path in _LOADED:
        return _LOADED[path]
    if path.is_dir():
        init = path / "__init__.py"
        if not init.is_file():
            raise FileNotFoundError(f"plugin package {path} has no __init__.py")
        name = f"rsf_plugins.{path.name}"
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(path)])
    elif path.suffix == ".py" and path.is_file():
        name = f"rsf_plugins.{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
    else:
        raise FileNotFoundError(f"plugin {path} not found")
    if "rsf_plugins" not in sys.modules:
        pkg = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("rsf_plugins", None, is_package=True))
        pkg.__path__ = []
        sys.modules["rsf_plugins"] = pkg
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _LOADED[path] = module
    return module

"""Feature selectors, as written in layouts and workflows.

A selector is one of:

* a string: a glob on the feature key, ``"incoming/L0/*"``;
* a mapping: every entry must match. Keys are ``key``, ``kind``, ``entity``,
  ``tags`` (all listed tags present), ``x``/``y``/``z``, or any attribute name.
  A string value with ``*?[`` is a glob; a list value means "one of"; a mapping
  value ``{min: a, max: b}`` is a closed range;
* a list of selectors: the union, in order, without duplicates.

The order of the result is the order of the input features (layout order); use
a scheduler or ``sort`` to reorder.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Iterable


class SelectorError(ValueError):
    pass


def _match_value(actual: Any, want: Any) -> bool:
    if isinstance(want, dict):
        unknown = set(want) - {"min", "max"}
        if unknown:
            raise SelectorError(f"range keys {sorted(unknown)}; use min/max")
        if actual is None:
            return False
        return ("min" not in want or actual >= want["min"]) and ("max" not in want or actual <= want["max"])
    if isinstance(want, (list, tuple)):
        return any(_match_value(actual, w) for w in want)
    if isinstance(want, str) and isinstance(actual, str) and any(c in want for c in "*?["):
        return fnmatch.fnmatchcase(actual, want)
    return actual == want


def matches(feature, selector) -> bool:
    if isinstance(selector, str):
        return fnmatch.fnmatchcase(feature.key, selector)
    if isinstance(selector, dict):
        for k, want in selector.items():
            if k == "tags":
                want = [want] if isinstance(want, str) else list(want)
                if not set(want) <= set(feature.tags):
                    return False
            elif not _match_value(feature.get(k), want):
                return False
        return True
    raise SelectorError(f"selector must be a string, mapping or list, got {type(selector).__name__}")


def select_features(features: Iterable, selector) -> list:
    features = list(features)
    if isinstance(selector, (list, tuple)):
        seen, out = set(), []
        for s in selector:
            for f in select_features(features, s):
                if f.key not in seen:
                    seen.add(f.key)
                    out.append(f)
        return out
    return [f for f in features if matches(f, selector)]


def sort_features(features: Iterable, keys: Iterable[str]) -> list:
    """Sort by attribute names; a leading ``-`` sorts that key descending."""
    out = list(features)
    for k in reversed(list(keys)):
        desc = k.startswith("-")
        name = k[1:] if desc else k
        out.sort(key=lambda f: f.get(name), reverse=desc)
    return out

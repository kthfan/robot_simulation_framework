"""Minimal URDF model: parse, forward kinematics, edit, write. numpy + stdlib only.

Enough for the framework's own needs (frame checks, collision geometry
extraction, composing robot + tool, compiling to other formats). Mesh paths are
resolved relative to the URDF file; ``package://`` URIs are resolved through an
optional ``packages`` map.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from rsf.core import se3

MOVABLE = ("revolute", "continuous", "prismatic")


def _floats(text: Optional[str], n: int, default=0.0) -> np.ndarray:
    if text is None:
        return np.full(n, float(default))
    vals = [float(v) for v in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return np.array(vals)


def _origin(el) -> np.ndarray:
    o = el.find("origin") if el is not None else None
    if o is None:
        return np.eye(4)
    return se3.from_xyz_rpy(_floats(o.get("xyz"), 3), _floats(o.get("rpy"), 3))


@dataclass
class Geometry:
    """One ``<visual>`` or ``<collision>`` element."""
    kind: str                       #: mesh | box | cylinder | sphere
    origin: np.ndarray              #: 4x4 in the link frame
    name: str = ""
    filename: Optional[str] = None  #: as written in the URDF
    path: Optional[Path] = None     #: resolved file, if found
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    size: Optional[np.ndarray] = None   #: box
    radius: Optional[float] = None      #: cylinder / sphere
    length: Optional[float] = None      #: cylinder


@dataclass
class Link:
    name: str
    visuals: list = field(default_factory=list)
    collisions: list = field(default_factory=list)
    mass: Optional[float] = None
    #: inertia tensor (3x3) about the centre of mass, in the frame of ``com``
    inertia: Optional[np.ndarray] = None
    com: np.ndarray = field(default_factory=lambda: np.eye(4))  #: inertial frame in the link frame


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    lower: float = -np.inf
    upper: float = np.inf
    velocity: Optional[float] = None
    effort: Optional[float] = None
    mimic: Optional[tuple] = None   #: (joint, multiplier, offset)

    @property
    def movable(self) -> bool:
        return self.type in MOVABLE

    def motion(self, q: float) -> np.ndarray:
        if self.type in ("revolute", "continuous"):
            return se3.make_T(se3.axis_angle(self.axis, q))
        if self.type == "prismatic":
            return se3.make_T(None, self.axis / np.linalg.norm(self.axis) * q)
        return np.eye(4)


class UrdfModel:
    """A parsed URDF as a tree of links and joints."""

    def __init__(self, name: str, links: dict, joints: dict, source: Optional[Path] = None):
        self.name = name
        self.links: dict[str, Link] = links
        self.joints: dict[str, Joint] = joints
        self.source = source
        self._reindex()

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_file(cls, path, packages: Optional[dict] = None) -> "UrdfModel":
        path = Path(path)
        return cls.from_string(path.read_text(), base_dir=path.parent, packages=packages, source=path)

    @classmethod
    def from_string(cls, text: str, base_dir=None, packages: Optional[dict] = None, source=None) -> "UrdfModel":
        root = ET.fromstring(text)
        base_dir = Path(base_dir) if base_dir is not None else None
        links, joints = {}, {}
        for el in root.findall("link"):
            link = Link(el.get("name"))
            for tag, dest in (("visual", link.visuals), ("collision", link.collisions)):
                for g in el.findall(tag):
                    geo = _geometry(g, base_dir, packages or {})
                    if geo is not None:
                        dest.append(geo)
            ine = el.find("inertial")
            if ine is not None and ine.find("mass") is not None:
                link.mass = float(ine.find("mass").get("value"))
                link.com = _origin(ine)
                i = ine.find("inertia")
                if i is not None:
                    v = {k: float(i.get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                    link.inertia = np.array([[v["ixx"], v["ixy"], v["ixz"]],
                                             [v["ixy"], v["iyy"], v["iyz"]],
                                             [v["ixz"], v["iyz"], v["izz"]]])
            links[link.name] = link
        for el in root.findall("joint"):
            lim = el.find("limit")
            mim = el.find("mimic")
            ax = el.find("axis")
            j = Joint(
                name=el.get("name"), type=el.get("type"),
                parent=el.find("parent").get("link"), child=el.find("child").get("link"),
                origin=_origin(el),
                axis=_floats(ax.get("xyz"), 3) if ax is not None else np.array([1.0, 0.0, 0.0]),
            )
            if lim is not None:
                j.lower = float(lim.get("lower", "-inf"))
                j.upper = float(lim.get("upper", "inf"))
                j.velocity = float(lim.get("velocity")) if lim.get("velocity") else None
                j.effort = float(lim.get("effort")) if lim.get("effort") else None
            if j.type == "continuous":
                j.lower, j.upper = -np.inf, np.inf
            if mim is not None:
                j.mimic = (mim.get("joint"), float(mim.get("multiplier", 1.0)), float(mim.get("offset", 0.0)))
            joints[j.name] = j
        return cls(root.get("name"), links, joints, source=Path(source) if source else None)

    def copy(self) -> "UrdfModel":
        return copy.deepcopy(self)

    def _reindex(self):
        self.child_joint = {j.child: j for j in self.joints.values()}
        self.children: dict[str, list] = {n: [] for n in self.links}
        for j in self.joints.values():
            self.children.setdefault(j.parent, []).append(j)
        roots = [n for n in self.links if n not in self.child_joint]
        if len(roots) != 1:
            raise ValueError(f"URDF {self.name!r} must have exactly one root link, has {roots}")
        self.root = roots[0]

    # ── queries ─────────────────────────────────────────────────────────────
    @property
    def movable_joints(self) -> list:
        """Movable, non-mimic joints in depth-first tree order."""
        out = []

        def walk(link):
            for j in self.children.get(link, []):
                if j.movable and j.mimic is None:
                    out.append(j.name)
                walk(j.child)

        walk(self.root)
        return out

    def chain(self, tip: str, base: Optional[str] = None) -> list:
        """Joint names from *base* (default root) down to *tip*."""
        base = base or self.root
        out, link = [], tip
        while link != base:
            if link not in self.child_joint:
                raise ValueError(f"{tip!r} is not below {base!r}")
            j = self.child_joint[link]
            out.append(j.name)
            link = j.parent
        return out[::-1]

    def limits(self, names) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.joints[n].lower for n in names]),
                np.array([self.joints[n].upper for n in names]))

    def _value(self, j: Joint, q: dict) -> float:
        if j.mimic is not None:
            src, mul, off = j.mimic
            return mul * q.get(src, 0.0) + off
        return q.get(j.name, 0.0)

    def fk(self, q=None, base: Optional[str] = None) -> dict:
        """Every link's pose in the *base* link frame (default root).

        *q*: ``{joint: value}``; missing joints are 0.
        """
        q = dict(q or {})
        out = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            link = stack.pop()
            for j in self.children.get(link, []):
                out[j.child] = out[link] @ j.origin @ j.motion(self._value(j, q))
                stack.append(j.child)
        if base is not None and base != self.root:
            Tb = se3.inv(out[base])
            out = {k: Tb @ v for k, v in out.items()}
        return out

    def link_pose(self, link: str, q=None, base: Optional[str] = None) -> np.ndarray:
        return self.fk(q, base)[link]

    # ── editing ─────────────────────────────────────────────────────────────
    def add_link(self, link: Link) -> None:
        if link.name in self.links:
            raise ValueError(f"link {link.name!r} exists")
        self.links[link.name] = link

    def add_joint(self, joint: Joint) -> None:
        if joint.name in self.joints:
            raise ValueError(f"joint {joint.name!r} exists")
        self.joints[joint.name] = joint
        self._reindex()

    def attach(self, other: "UrdfModel", parent_link: str, origin=None, joint_name=None, prefix: str = "") -> None:
        """Graft *other* under *parent_link* with a fixed joint at *origin*."""
        other = other.copy()
        ren = {n: prefix + n for n in other.links}
        for n, link in other.links.items():
            link.name = ren[n]
            self.add_link(link)
        for j in other.joints.values():
            j.name, j.parent, j.child = prefix + j.name, ren[j.parent], ren[j.child]
            if j.mimic is not None:
                j.mimic = (prefix + j.mimic[0],) + tuple(j.mimic[1:])
            self.joints[j.name] = j
        self.add_joint(Joint(joint_name or f"{parent_link}_to_{ren[other.root]}", "fixed",
                             parent_link, ren[other.root], np.eye(4) if origin is None else np.asarray(origin, dtype=float)))

    def remove_link(self, name: str) -> None:
        """Remove a leaf link and the joint above it."""
        if self.children.get(name):
            raise ValueError(f"{name!r} is not a leaf")
        j = self.child_joint.get(name)
        if j is not None:
            del self.joints[j.name]
        del self.links[name]
        self._reindex()

    # ── output ──────────────────────────────────────────────────────────────
    def to_xml(self, mesh_dir: Optional[Path] = None, relative_to: Optional[Path] = None) -> str:
        """URDF text. Mesh paths are written relative to *relative_to* when given."""
        root = ET.Element("robot", name=self.name)
        for link in self.links.values():
            el = ET.SubElement(root, "link", name=link.name)
            if link.mass is not None:
                ine = ET.SubElement(el, "inertial")
                _write_origin(ine, link.com)
                ET.SubElement(ine, "mass", value=_fmt(link.mass))
                I = link.inertia if link.inertia is not None else np.eye(3) * 1e-4
                ET.SubElement(ine, "inertia", ixx=_fmt(I[0, 0]), ixy=_fmt(I[0, 1]), ixz=_fmt(I[0, 2]),
                              iyy=_fmt(I[1, 1]), iyz=_fmt(I[1, 2]), izz=_fmt(I[2, 2]))
            for tag, geos in (("visual", link.visuals), ("collision", link.collisions)):
                for g in geos:
                    ge = ET.SubElement(el, tag, **({"name": g.name} if g.name else {}))
                    _write_origin(ge, g.origin)
                    gg = ET.SubElement(ET.SubElement(ge, "geometry"), g.kind)
                    if g.kind == "mesh":
                        fn = g.path if g.path is not None else Path(g.filename)
                        if relative_to is not None and g.path is not None:
                            fn = Path(_relpath(g.path, relative_to))
                        gg.set("filename", fn.as_posix())
                        if not np.allclose(g.scale, 1.0):
                            gg.set("scale", _fmtv(g.scale))
                    elif g.kind == "box":
                        gg.set("size", _fmtv(g.size))
                    elif g.kind == "cylinder":
                        gg.set("radius", _fmt(g.radius))
                        gg.set("length", _fmt(g.length))
                    elif g.kind == "sphere":
                        gg.set("radius", _fmt(g.radius))
        for j in self.joints.values():
            el = ET.SubElement(root, "joint", name=j.name, type=j.type)
            _write_origin(el, j.origin)
            ET.SubElement(el, "parent", link=j.parent)
            ET.SubElement(el, "child", link=j.child)
            if j.type != "fixed":
                ET.SubElement(el, "axis", xyz=_fmtv(j.axis))
            if j.type in ("revolute", "prismatic"):
                lim = {"lower": _fmt(j.lower), "upper": _fmt(j.upper),
                       "effort": _fmt(j.effort if j.effort is not None else 1000.0),
                       "velocity": _fmt(j.velocity if j.velocity is not None else 1.0)}
                ET.SubElement(el, "limit", **lim)
            if j.mimic is not None:
                ET.SubElement(el, "mimic", joint=j.mimic[0], multiplier=_fmt(j.mimic[1]), offset=_fmt(j.mimic[2]))
        ET.indent(root)
        return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"

    def write(self, path) -> None:
        path = Path(path)
        path.write_text(self.to_xml(relative_to=path.parent))


def _relpath(p: Path, start: Path) -> str:
    import os
    return os.path.relpath(Path(p).resolve(), Path(start).resolve())


def _fmt(v: float) -> str:
    return repr(float(v))


def _fmtv(v) -> str:
    return " ".join(_fmt(x) for x in v)


def _write_origin(el, T) -> None:
    T = np.asarray(T, dtype=float)
    if np.allclose(T, np.eye(4), atol=0.0):
        return
    ET.SubElement(el, "origin", xyz=_fmtv(T[:3, 3]), rpy=_fmtv(se3.matrix_to_rpy(T[:3, :3])))


def _geometry(el, base_dir: Optional[Path], packages: dict) -> Optional[Geometry]:
    g = el.find("geometry")
    if g is None or len(g) == 0:
        return None
    shape = g[0]
    geo = Geometry(kind=shape.tag, origin=_origin(el), name=el.get("name") or "")
    if shape.tag == "mesh":
        geo.filename = shape.get("filename")
        geo.scale = _floats(shape.get("scale"), 3, 1.0)
        geo.path = resolve_uri(geo.filename, base_dir, packages)
    elif shape.tag == "box":
        geo.size = _floats(shape.get("size"), 3)
    elif shape.tag == "cylinder":
        geo.radius, geo.length = float(shape.get("radius")), float(shape.get("length"))
    elif shape.tag == "sphere":
        geo.radius = float(shape.get("radius"))
    else:
        raise ValueError(f"unsupported geometry {shape.tag!r}")
    return geo


def resolve_uri(uri: str, base_dir: Optional[Path], packages: dict) -> Optional[Path]:
    """File path of a URDF mesh URI, or None if it cannot be found."""
    if uri.startswith("package://"):
        pkg, _, rest = uri[len("package://"):].partition("/")
        if pkg not in packages:
            return None
        p = Path(packages[pkg]) / rest
    elif uri.startswith("file://"):
        p = Path(uri[len("file://"):])
    else:
        p = Path(uri)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
    return p.resolve() if p.exists() else None

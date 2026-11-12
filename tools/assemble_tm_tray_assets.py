"""Assemble the TM tray cell's robot, gripper and object assets from their sources.

What goes where is declared in ``cells/tm_tray/asset_spec.yaml``; this script
only carries it out. ``--sources`` is a directory laid out as the spec's paths
expect::

    sources/
      tm_description/   TechmanRobotInc/tm2_ros2 (humble) tm_description
      v13/              tm5s_v13.urdf + meshes/   (user, exported from tm5s.usd)
      egu50/            egu50_v2.urdf + meshs/    (AI-Robotics-V2 program/robot_config, D29)
      jig/              tray_jig_with_label.urdf + meshes/ (user)

Only collision meshes exist for the user-supplied parts, so they double as
visuals; the arm also uses tm_description's collision STLs as visuals.
Provenance and the evidence for each choice: docs/assets.md. Run::

    python tools/assemble_tm_tray_assets.py --sources <dir>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsf.core import se3  # noqa: E402
from rsf.model import primitives  # noqa: E402
from rsf.model.urdf import Geometry, Joint, Link, UrdfModel  # noqa: E402

CELL = ROOT / "cells" / "tm_tray"


def mesh(path: Path, origin=None, name: str = "") -> Geometry:
    return Geometry("mesh", np.eye(4) if origin is None else np.asarray(origin, float), name=name,
                    filename=str(path), path=path.resolve())


def copy_mesh(src: Path, dst_dir: Path, name: str | None = None) -> Path:
    """Copy a mesh; OBJ material references are dropped (no .mtl is shipped)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or src.name)
    if src.suffix.lower() == ".obj":
        lines = [ln for ln in src.read_text().splitlines() if not ln.startswith(("mtllib", "usemtl"))]
        dst.write_text("\n".join(lines) + "\n")
    else:
        shutil.copyfile(src, dst)
    return dst


def build_arm(spec: dict, src: Path, out: Path) -> UrdfModel:
    """The calibrated chain of ``chain_urdf`` with ``link_meshes`` as its geometry."""
    chain = UrdfModel.from_file(src / spec["chain_urdf"])
    mdir = out / spec["name"] / "meshes"
    mesh_src = src / spec["mesh_source"]
    (out / spec["name"]).mkdir(parents=True, exist_ok=True)
    if spec.get("license_file"):
        shutil.copyfile(src / spec["license_file"], out / spec["name"] / "LICENSE.mesh_source")
    links = {"base": Link("base")}
    for name, f in spec["link_meshes"].items():
        g = mesh(copy_mesh(mesh_src / f, mdir), name=name)
        s = chain.links[name]
        links[name] = Link(name, visuals=[g], collisions=[g], mass=s.mass, inertia=s.inertia, com=s.com.copy())
    tip = spec["tip_link"]
    links[tip] = Link(tip)
    joints = {"base_fixed_joint": Joint("base_fixed_joint", "fixed", "base", "link_0", np.eye(4))}
    for jn in chain.chain(tip, base="link_0"):
        j = chain.joints[jn]
        joints[jn] = Joint(jn, j.type, j.parent, j.child, j.origin.copy(), j.axis.copy(),
                           j.lower, j.upper, j.velocity, spec.get("effort", {}).get(jn, j.effort))
    return UrdfModel(spec["name"], links, joints)


def build_tool(spec: dict, src: Path, out: Path, options: list, chain: UrdfModel) -> UrdfModel:
    """The gripper: ``urdf``'s joints and body colliders, ``jaws`` colliders, ``options``."""
    egu = UrdfModel.from_file(src / spec["urdf"])
    egu_dir = (src / spec["urdf"]).parent
    v13_dir = src / "v13" / "meshes"
    mdir = out / spec["name"] / "meshes"
    body = egu.links["gripper_body"]
    geos = []
    for g in body.collisions:
        part = spec["body_parts"].get(Path(g.filename).name)
        if part is not None:
            geos.append(mesh(copy_mesh(egu_dir / "meshs" / Path(g.filename).name, mdir, f"{part}.obj"),
                             g.origin, part))
    missing = set(spec["body_parts"].values()) - {g.name for g in geos}
    if missing:
        raise SystemExit(f"{spec['urdf']}: colliders not found for {sorted(missing)}")

    mount = se3.from_xyz_rpy(spec["mount"]["xyz"], spec["mount"]["rpy"])
    for opt in options:
        o = spec["options"][opt]
        # Re-express colliders given in a robot link frame in the gripper frame.
        T = np.eye(4)
        if o["frame"] != spec["mount"]["parent"]:
            T = se3.inv(chain.link_pose(spec["mount"]["parent"], base=o["frame"]))
        T = se3.inv(mount) @ T
        for part, f in o["parts"].items():
            geos.append(mesh(copy_mesh(v13_dir / f, mdir, f"{part}.obj"), T, part))

    links = {"gripper_base": Link("gripper_base"),
             "gripper_body": Link("gripper_body", visuals=list(geos), collisions=list(geos),
                                  mass=body.mass, inertia=body.inertia, com=body.com.copy())}
    joints = {"gripper_fixed": Joint("gripper_fixed", "fixed", "gripper_base", "gripper_body", np.eye(4))}
    for jaw, files in spec["jaws"].items():
        gs = [mesh(copy_mesh(v13_dir / f, mdir, f"{jaw}_{k}.obj"), name=f"{jaw}_{k}") for k, f in enumerate(files)]
        s = egu.links[jaw]
        links[jaw] = Link(jaw, visuals=list(gs), collisions=list(gs), mass=s.mass, inertia=s.inertia, com=s.com.copy())
        j = egu.joints[f"{jaw}_joint"]
        joints[j.name] = Joint(j.name, "prismatic", "gripper_body", jaw, j.origin.copy(), j.axis.copy(),
                               j.lower, j.upper, j.velocity, 100.0)
    name = spec["name"] + "".join(f"_{o}" for o in options)
    return UrdfModel(name, links, joints)


def build_object(name: str, spec: dict, src: Path, out: Path) -> UrdfModel:
    mdir = out / name / "meshes"
    if "urdf" in spec:
        s = UrdfModel.from_file(src / spec["urdf"])
        geos = [mesh(copy_mesh(g.path, mdir), g.origin, Path(g.filename).stem)
                for link in s.links.values() for g in link.visuals + link.collisions]
        return UrdfModel(name, {name: Link(name, visuals=list(geos), collisions=list(geos), mass=spec.get("mass"))}, {})
    boxes = getattr(primitives, spec["generator"])(**spec["params"])
    return UrdfModel(name, {name: primitives.box_link(name, boxes, mdir, spec.get("mass"))}, {})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=CELL / "asset_spec.yaml")
    ap.add_argument("--out", type=Path, default=CELL / "assets")
    a = ap.parse_args(argv)
    spec = yaml.safe_load(a.spec.read_text())
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)

    chain = UrdfModel.from_file(a.sources / spec["robot"]["chain_urdf"])
    robot = build_arm(spec["robot"], a.sources, a.out)
    robot.write(a.out / robot.name / f"{robot.name}.urdf")
    tool = spec["tool"]
    mount = se3.from_xyz_rpy(tool["mount"]["xyz"], tool["mount"]["rpy"])
    for fname, options in spec["variants"].items():
        g = build_tool(tool, a.sources, a.out, list(options), chain)
        g.write(a.out / tool["name"] / f"{g.name}.urdf")
        full = robot.copy()
        full.name = Path(fname).stem
        full.attach(g, tool["mount"]["parent"], mount, joint_name="flange_to_gripper")
        full.write(a.out / fname)
    for name, ospec in spec["objects"].items():
        build_object(name, ospec, a.sources, a.out).write(a.out / name / f"{name}.urdf")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

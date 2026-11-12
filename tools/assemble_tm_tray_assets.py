"""Assemble the TM tray cell's robot, gripper and object assets from their sources.

Sources (downloaded separately, paths passed on the command line):

* ``--tm-description``: TechmanRobotInc/tm2_ros2 (humble) ``tm_description``.
  Arm link collision meshes (BSD-3-Clause, see LICENSE copied next to them).
* ``--v13``: ``tm5s_v13.urdf`` and its ``meshes/`` (user, exported from tm5s.usd).
  Calibrated joint chain, jaw colliders, air-cylinder colliders.
* ``--egu50``: the D29-corrected gripper (``egu50_v2.urdf`` + ``meshs/``) from
  AI-Robotics-V2 ``program/robot_config``. Body, plate and D455 colliders and the
  jaw joints, without tm5s.usd's stray (0, +3.41, -1.76) mm mount offset.
* ``--jig``: ``tray_jig_with_label.urdf`` and ``meshes/`` (user).

Only collision meshes exist for the user-supplied parts, so they are also used as
visuals. The arm uses tm_description's collision STLs as visuals too (its visual
OBJs are ~40 MB).

Decisions recorded in docs/assets.md. Run::

    python tools/assemble_tm_tray_assets.py --tm-description ... --v13 ... --egu50 ... --jig ...
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsf.core import se3  # noqa: E402
from rsf.model.urdf import Geometry, Joint, Link, UrdfModel  # noqa: E402

OUT = ROOT / "cells" / "tm_tray" / "assets"

#: tm_description collision mesh per arm link.
ARM_MESHES = {
    "link_0": "tm5s-base_c.stl", "link_1": "tm5s-j1_c.stl", "link_2": "tm5s-a1_c.stl",
    "link_3": "tm5s-a2_c.stl", "link_4": "tm5s-j5_c.stl", "link_5": "tm5s-j6_c.stl",
    "link_6": "tm5s-io_c.stl",
}
#: tm_description effort limits (N m); v13 carries none.
ARM_EFFORT = {"joint_1": 157.0, "joint_2": 157.0, "joint_3": 157.0,
              "joint_4": 54.0, "joint_5": 54.0, "joint_6": 34.0}

#: Gripper base frame in the flange frame. The gripper's -y is the approach
#: (flange +z) and its +z is flange +y ("up", the D455 side).
FLANGE_T_GRIPPER = se3.make_T(se3.rot_x(-np.pi / 2))

#: Tray outer size (x width, y length, z height), user 2026-11-12:
#: "tray的dim為0.25, 0.13, 0.9" read as 0.25 long, 0.13 wide, 0.09 high.
TRAY_SIZE = (0.13, 0.25, 0.09)


def mesh(path: Path, origin=None, name: str = "") -> Geometry:
    return Geometry("mesh", np.eye(4) if origin is None else origin, name=name,
                    filename=str(path), path=path.resolve())


def copy(src: Path, dst_dir: Path, name: str | None = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or src.name)
    shutil.copyfile(src, dst)
    return dst


def strip_mtl(path: Path) -> None:
    """Drop ``mtllib``/``usemtl`` lines: the .mtl files are not shipped."""
    lines = [ln for ln in path.read_text().splitlines()
             if not ln.startswith(("mtllib", "usemtl"))]
    path.write_text("\n".join(lines) + "\n")


def box_obj(path: Path, lo, hi, comment: str) -> Path:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = [(x, y, z) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    faces = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    tris = [t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {comment}\n")
        for p in v:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*p))
        for t in tris:
            f.write("f {} {} {}\n".format(*t))
    return path


def arm(tm_desc: Path, v13: UrdfModel) -> UrdfModel:
    """v13's calibrated chain, fixed to the floor, with tm_description's meshes."""
    mdir = OUT / "tm5s" / "meshes"
    for f in ARM_MESHES.values():
        copy(tm_desc / "meshes" / "tm5s" / "collision" / f, mdir)
    lic = next(iter(sorted(tm_desc.parent.glob("LICENSE*"))), None)
    if lic is not None:
        copy(lic, OUT / "tm5s", "LICENSE.tm_description")
    links = {"base": Link("base")}
    for name, f in ARM_MESHES.items():
        src = v13.links[name]
        g = mesh(mdir / f, name=name)
        links[name] = Link(name, visuals=[g], collisions=[g], mass=src.mass,
                           inertia=src.inertia, com=src.com.copy())
    links["flange"] = Link("flange")
    joints = {"base_fixed_joint": Joint("base_fixed_joint", "fixed", "base", "link_0", np.eye(4))}
    for i in range(1, 7):
        j = v13.joints[f"joint_{i}"]
        joints[j.name] = Joint(j.name, "revolute", j.parent, j.child, j.origin.copy(), j.axis.copy(),
                               j.lower, j.upper, j.velocity, ARM_EFFORT[j.name])
    fj = v13.joints["flange_fixed_joint"]
    joints[fj.name] = Joint(fj.name, "fixed", "link_6", "flange", fj.origin.copy())
    return UrdfModel("tm5s", links, joints)


def gripper(egu: UrdfModel, egu_dir: Path, v13_dir: Path, air_cylinder: bool, v13: UrdfModel) -> UrdfModel:
    """The D29 gripper: egu50 body/plate/D455 and joints, v13's jaw colliders."""
    mdir = OUT / "egu50" / "meshes"
    body = egu.links["gripper_body"]
    want = {"gripper_body_collisions_Cube.obj": "body", "gripper_body_collisions_Cube_01.obj": "plate",
            "d455_collisions_Cube.obj": "d455"}
    geos = []
    for g in body.collisions:
        fname = Path(g.filename).name
        if fname not in want:
            continue
        p = copy(egu_dir / "meshs" / fname, mdir, f"{want[fname]}.obj")
        geos.append(mesh(p, g.origin, want[fname]))
    if len(geos) != 3:
        raise SystemExit(f"egu50: expected body, plate, d455 colliders, got {[g.name for g in geos]}")
    if air_cylinder:
        # v13 places them in link_6; re-express in the gripper frame. Whether the
        # air cylinder also carries the stray mount offset is unknown: kept as is.
        T = se3.inv(v13.joints["flange_fixed_joint"].origin @ FLANGE_T_GRIPPER)
        for src, nm in (("Cube_1.obj", "air_cylinder_plate"), ("Cube_01_1.obj", "air_cylinder_body")):
            p = copy(v13_dir / "meshes" / src, mdir, f"{nm}.obj")
            strip_mtl(p)
            geos.append(mesh(p, T, nm))
    links = {"gripper_base": Link("gripper_base"),
             "gripper_body": Link("gripper_body", visuals=list(geos), collisions=list(geos),
                                  mass=body.mass, inertia=body.inertia, com=body.com.copy())}
    joints = {"gripper_fixed": Joint("gripper_fixed", "fixed", "gripper_base", "gripper_body", np.eye(4))}
    for jaw, files in (("right_gripper", ("Cube_2.obj", "Cube_01_2.obj")),
                       ("left_gripper", ("Cube_3.obj", "Cube_01_3.obj"))):
        gs = []
        for k, src in enumerate(files):
            p = copy(v13_dir / "meshes" / src, mdir, f"{jaw}_{k}.obj")
            strip_mtl(p)
            gs.append(mesh(p, name=f"{jaw}_{k}"))
        src = egu.links[jaw]
        links[jaw] = Link(jaw, visuals=list(gs), collisions=list(gs),
                          mass=src.mass, inertia=src.inertia, com=src.com.copy())
        j = egu.joints[f"{jaw}_joint"]
        joints[j.name] = Joint(j.name, "prismatic", "gripper_body", jaw, j.origin.copy(), j.axis.copy(),
                               j.lower, j.upper, j.velocity, 100.0)
    for f in mdir.glob("*.obj"):
        strip_mtl(f)
    return UrdfModel("egu50" + ("_air_cylinder" if air_cylinder else ""), links, joints)


def jig(jig_dir: Path) -> UrdfModel:
    src = UrdfModel.from_file(jig_dir / "tray_jig_with_label.urdf")
    mdir = OUT / "tray_jig_with_label" / "meshes"
    geos = []
    for link in src.links.values():
        for g in link.visuals + link.collisions:
            p = copy(g.path, mdir)
            strip_mtl(p)
            geos.append(mesh(p, g.origin, Path(g.filename).stem))
    return UrdfModel("tray_jig_with_label",
                     {"jig": Link("jig", visuals=list(geos), collisions=list(geos))}, {})


def tray() -> UrdfModel:
    """A box, frame at the centre of its underside."""
    x, y, z = TRAY_SIZE
    p = box_obj(OUT / "tray" / "meshes" / "tray.obj", (-x / 2, -y / 2, 0.0), (x / 2, y / 2, z),
                f"tray {x} x {y} x {z} m (x width, y length, z height); frame at the underside centre")
    g = mesh(p, name="tray")
    return UrdfModel("tray", {"tray": Link("tray", visuals=[g], collisions=[g], mass=1.2)}, {})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tm-description", type=Path, required=True)
    ap.add_argument("--v13", type=Path, required=True, help="dir with tm5s_v13.urdf and meshes/")
    ap.add_argument("--egu50", type=Path, required=True, help="dir with egu50_v2.urdf and meshs/")
    ap.add_argument("--jig", type=Path, required=True, help="dir with tray_jig_with_label.urdf and meshes/")
    a = ap.parse_args(argv)

    if OUT.exists():
        shutil.rmtree(OUT)
    v13 = UrdfModel.from_file(a.v13 / "tm5s_v13.urdf")
    egu = UrdfModel.from_file(a.egu50 / "egu50_v2.urdf")

    robot = arm(a.tm_description, v13)
    robot.write(OUT / "tm5s" / "tm5s.urdf")
    for air in (False, True):
        g = gripper(egu, a.egu50, a.v13, air, v13)
        g.write(OUT / "egu50" / f"{g.name}.urdf")
        full = robot.copy()
        full.name = f"tm5s_{g.name}"
        full.attach(g, "flange", FLANGE_T_GRIPPER, joint_name="flange_to_gripper")
        full.write(OUT / f"tm5s_{g.name}.urdf")
    jig(a.jig).write(OUT / "tray_jig_with_label" / "tray_jig_with_label.urdf")
    tray().write(OUT / "tray" / "tray.urdf")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

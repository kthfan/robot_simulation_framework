"""P5: the world as a USD stage for Isaac Sim (``rsf.compile.usd``), checked
against the URDF and against the MuJoCo model of the same world."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("trimesh")

from rsf.compile import usd as U  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402
from rsf.model.urdf import UrdfModel  # noqa: E402
from rsf.sim.isaac_cell import IsaacParams  # noqa: E402

CELL = Path(__file__).resolve().parents[1] / "cells" / "tm_tray"


@pytest.fixture(scope="module")
def cell():
    world = load_world(CELL / "layouts" / "strict.yaml")
    return world, U.world_stage(world), UrdfModel.from_file(world.robot.urdf)


def _world_T(prim):
    from pxr import Usd, UsdGeom
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def _by_kind(stage, kind):
    return {p.GetCustomDataByKey("rsf:name"): p for p in stage.Traverse() if p.GetCustomDataByKey("rsf:kind") == kind}


def test_articulation_matches_the_urdf(cell):
    """The stage's joints (frames, axes, prismatic vs revolute) reproduce the URDF's FK."""
    world, stage, m = cell
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = {n: rng.uniform(0.01, 0.06) if m.joints[n].type == "prismatic" else rng.uniform(-3, 3)
             for n in m.movable_joints}
        P = U.stage_fk(stage, q)
        fk = m.fk(q)
        assert set(P) == {"base", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
                          "left_gripper", "right_gripper"}
        for b, T in P.items():
            assert np.abs(world.robot.base @ fk[b] - T).max() < 1e-6, b


def test_bodies_are_authored_where_the_joints_put_them(cell):
    """At q = 0 every body's prim sits where its joint chain puts it (PhysX starts there);
    the flange is a frame on link_6 where the URDF has it."""
    world, stage, m = cell
    P = U.stage_fk(stage, {})
    bodies = _by_kind(stage, "robot_body")
    for b, T in P.items():
        assert np.abs(_world_T(bodies[b]) - T).max() < 1e-6, b
    body, T_bl = U.link_map(stage)["flange"]
    assert body == "link_6"
    assert np.abs(P["link_6"] @ T_bl - world.robot.base @ m.fk()["flange"]).max() < 1e-6


def test_drives_limits_and_gravity(cell):
    from pxr import UsdPhysics
    world, stage, m = cell
    p = IsaacParams()
    joints = _by_kind(stage, "robot_joint")
    assert set(joints) == {"root_fixed", *m.movable_joints}
    for k, n in enumerate(f"joint_{i}" for i in range(1, 7)):
        J = UsdPhysics.RevoluteJoint(joints[n])
        d = UsdPhysics.DriveAPI(joints[n], "angular")
        assert d.GetStiffnessAttr().Get() == pytest.approx(p.arm_kp[k] * np.pi / 180)     # per degree
        assert d.GetDampingAttr().Get() == pytest.approx(p.arm_kv[k] * np.pi / 180)
        lo, hi = m.joints[n].lower, m.joints[n].upper
        assert J.GetLowerLimitAttr().Get() == pytest.approx(np.degrees(lo), abs=1e-4)
        assert J.GetUpperLimitAttr().Get() == pytest.approx(np.degrees(hi), abs=1e-4)
    for n in ("left_gripper_joint", "right_gripper_joint"):
        d = UsdPhysics.DriveAPI(joints[n], "linear")
        assert d.GetMaxForceAttr().Get() == pytest.approx(p.jaw_force)
        assert UsdPhysics.PrismaticJoint(joints[n]).GetLowerLimitAttr().Get() == pytest.approx(m.joints[n].lower)
    for b in _by_kind(stage, "robot_body").values():
        assert b.GetAttribute("physxRigidBody:disableGravity").Get() is True
    root = stage.GetPrimAtPath(f"{U.ROOT}/robot")
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is False


def test_static_solids_and_objects_where_the_world_has_them(cell):
    world, stage, m = cell
    statics = _by_kind(stage, "static")
    solids = world.solids(objects=False)
    assert len(statics) == len(solids) == 504
    for s in solids[::7]:
        M = _world_T(statics[s.name])                      # unit cube scaled to the box
        corners = np.array([[x, y, z, 1] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) @ M.T
        assert np.allclose(np.sort(corners[:, :3], axis=0), np.sort(s.corners(), axis=0), atol=1e-9), s.name
    objs = _by_kind(stage, "object")
    assert len(objs) == len(world.objects) == 37
    for o in world.objects[::5]:
        assert np.abs(_world_T(objs[o.name]) - o.T).max() < 1e-9


def test_mass_properties_match_mujoco(cell):
    """Masses and centres of mass as MuJoCo computes them for the same world (objects by
    volume from their boxes; robot links merged across fixed joints)."""
    mujoco = pytest.importorskip("mujoco")
    from pxr import UsdPhysics
    from rsf.compile.mjcf import ROBOT_PREFIX, world_spec
    world, stage, m = cell
    mm = world_spec(world, actuators=False).compile()
    d = mujoco.MjData(mm)
    mujoco.mj_forward(mm, d)
    for o in world.objects[::6]:
        api = UsdPhysics.MassAPI(_by_kind(stage, "object")[o.name])
        b = mm.body(o.name)
        assert api.GetMassAttr().Get() == pytest.approx(b.mass[0])
        assert np.allclose(api.GetCenterOfMassAttr().Get(), b.ipos, atol=1e-6)
        assert np.allclose(sorted(api.GetDiagonalInertiaAttr().Get()), sorted(b.inertia), rtol=1e-5)
    groups = U.robot_bodies(m)
    for body, prim in _by_kind(stage, "robot_body").items():
        links = [ln for ln, (bb, _) in groups.items() if bb == body]
        mass = sum(mm.body(ROBOT_PREFIX + ln).mass[0] for ln in links)
        if mass == 0:
            continue
        api = UsdPhysics.MassAPI(prim)
        assert api.GetMassAttr().Get() == pytest.approx(mass, rel=1e-6), body
        com = sum(mm.body(ROBOT_PREFIX + ln).mass[0] * d.xipos[mm.body(ROBOT_PREFIX + ln).id] for ln in links) / mass
        assert np.allclose(_world_T(prim) @ np.r_[api.GetCenterOfMassAttr().Get(), 1.0], np.r_[com, 1], atol=1e-6), body


def test_names_survive_and_materials(cell):
    from pxr import UsdPhysics, UsdShade
    world, stage, m = cell
    p = IsaacParams()
    cols = [q for q in stage.Traverse() if q.GetCustomDataByKey("rsf:kind") == "robot_collision"]
    assert {c.GetCustomDataByKey("rsf:name") for c in cols} >= {"gripper_body", "link_6", "left_gripper"}
    for c in cols:
        assert c.GetAttribute("physics:approximation").Get() == "convexHull"
    for k in ("robot", "object", "static"):
        mat = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f"{U.ROOT}/materials/{k}"))
        assert mat.GetStaticFrictionAttr().Get() == pytest.approx(p.friction[k])
        assert mat.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"
    tray = next(q for q in stage.Traverse() if q.GetCustomDataByKey("rsf:kind") == "object_solid")
    bound, _ = UsdShade.MaterialBindingAPI(tray).ComputeBoundMaterial("physics")
    assert bound.GetPath().pathString == f"{U.ROOT}/materials/object"


def test_round_trip_through_a_file(cell, tmp_path):
    from pxr import Usd
    world, _, m = cell
    path = tmp_path / "cell.usda"
    U.world_stage(world, path)
    st = Usd.Stage.Open(str(path))
    q = {"joint_2": 0.7, "joint_5": -1.1, "left_gripper_joint": 0.03}
    P = U.stage_fk(st, q)
    assert np.abs(P["link_6"] - world.robot.base @ m.fk(q)["link_6"]).max() < 1e-6


def test_isaac_cell_maps_names_without_isaac(tmp_path):
    """The Isaac cell compiles its stage and maps PhysX's paths back to the judge's names
    before Isaac starts (contacts: robot link vs object or static solid; the flange)."""
    from rsf.model.workflow import Workflow
    from rsf.sim.isaac_cell import IsaacCell, _T7
    world = load_world(CELL / "layouts" / "strict.yaml")
    wf = Workflow.from_file(CELL / "workflows" / "inspect_and_sort.yaml")
    occ = wf.initial_occupancy(world, wf.resolve_sets(world))
    cell = IsaacCell(world, occ, usd_path=tmp_path / "c.usda")
    assert len(cell.object_names) == len(occ) < len(world.objects)
    o = cell.object_names[0]
    assert cell._who(cell.object_paths[o] + "/body") == (False, o)
    assert cell._who(f"{U.ROOT}/robot/link_6/gripper_body/collision_1") == (True, "gripper_body")
    s = world.solids(objects=False)[3]
    assert cell._who(f"{U.ROOT}/static/{U.prim_name(s.name)}") == (False, s.name)
    # flange from PhysX's link poses ([p, q_xyzw] per body, in the articulation's link order)
    from pxr import Usd
    st = Usd.Stage.Open(str(cell.usd_path))
    q = {"joint_1": 0.4, "joint_3": -1.2, "joint_6": 2.0}
    P = U.stage_fk(st, q)
    cell.link_names = [U.prim_name(b) for b in P]
    from rsf.core import se3
    rows = [np.r_[T[:3, 3], np.roll(se3.quat_wxyz(T[:3, :3]), -1)] for T in P.values()]
    assert np.abs(_T7(rows[0]) - list(P.values())[0]).max() < 1e-12
    want = world.robot.base @ cell.urdf.link_pose("flange", q)
    assert np.abs(cell._flange_actual(np.array(rows)) - want).max() < 1e-6

"""USD models in layouts (rsf.model.usd_model): the same parts as the URDFs, written as
USD in the ways real files come (centimetres, Y up, nested transforms, instancing,
physics-marked colliders next to visual meshes), must give the same World."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
trimesh = pytest.importorskip("trimesh")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from rsf.core import se3  # noqa: E402
from rsf.model import audit as AU  # noqa: E402
from rsf.model import usd_model as UM  # noqa: E402
from rsf.model.layout import LayoutError, build_world, load_document  # noqa: E402
from rsf.model.urdf import UrdfModel  # noqa: E402
from rsf.model.world import solids_from_urdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "cells" / "tm_tray"
JIG = TM / "assets" / "tray_jig_with_label" / "tray_jig_with_label.urdf"
TRAY = TM / "assets" / "tray" / "tray.urdf"
EXAMPLE = ROOT / "cells" / "example_pick_place"


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RSF_CACHE", str(tmp_path / "cache"))


# ── writing USD the way exporters do ────────────────────────────────────────
def _parts(urdf) -> list:
    """``(name, T, V, F)`` of every collision mesh of a one-link URDF, in its frame."""
    m = UrdfModel.from_file(urdf)
    out = []
    for link in m.links.values():
        for i, g in enumerate(link.collisions):
            tm = trimesh.load(g.path, force="mesh", process=False)
            out.append((g.name or f"g{i}", g.origin, np.asarray(tm.vertices), np.asarray(tm.faces)))
    return out


def _xform(prim_or_geom, M) -> None:
    """Set a prim's transform to the column-convention 4x4 *M*."""
    UsdGeom.Xformable(prim_or_geom).AddTransformOp().Set(Gf.Matrix4d(np.asarray(M, float).T.tolist()))


def _mesh(stage, path, V, F):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in V])
    m.CreateFaceVertexCountsAttr([3] * len(F))
    m.CreateFaceVertexIndicesAttr([int(i) for i in np.asarray(F).ravel()])
    return m


def write_usd(path, parts, *, mpu=1.0, up="Z", wrap=np.eye(4), root="/part", physics=False, decoy=False,
              mass=None, kpu=1.0):
    """*parts* (metres, z up, in the part frame) as a USD stage in *mpu* units and *up* axis.
    *wrap*: an extra transform on a group Xform, undone in the parts (nested transforms);
    *physics*: colliders get CollisionAPI; *decoy*: a large visual-only mesh beside them."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
    if kpu != 1.0:
        UsdPhysics.SetStageKilogramsPerUnit(stage, kpu)
    # rsf (metres, z up) -> stage coordinates
    to_stage = np.diag([1 / mpu] * 3 + [1.0])
    if up == "Y":
        to_stage = to_stage @ se3.make_T(se3.rot_x(-math.pi / 2))
    r = UsdGeom.Xform.Define(stage, root)
    stage.SetDefaultPrim(r.GetPrim())
    g = UsdGeom.Xform.Define(stage, f"{root}/geo")
    _xform(g, wrap)
    for name, T, V, F in parts:
        # the part's pose and points in stage coordinates, under the wrap
        M = np.linalg.inv(wrap) @ to_stage @ np.asarray(T) @ np.linalg.inv(to_stage)
        m = _mesh(stage, f"{root}/geo/{name}", (V / mpu) @ (to_stage[:3, :3] * mpu).T, F)
        _xform(m, M)
        if physics:
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    if decoy:
        big = trimesh.creation.box(extents=(5, 5, 5))
        _mesh(stage, f"{root}/visual_only", np.asarray(big.vertices) / mpu, np.asarray(big.faces))
    if mass is not None:
        UsdPhysics.MassAPI.Apply(r.GetPrim()).CreateMassAttr(mass)
    stage.GetRootLayer().Save()
    return Path(path)


def _key(s):
    return tuple(np.round(np.asarray(s.T)[:3, 3], 6)) + (s.shape.kind,)


def _same_solids(a, b, tol=1e-6):
    """Two solid lists equal as geometry (names may differ)."""
    assert len(a) == len(b)
    for x, y in zip(sorted(a, key=_key), sorted(b, key=_key)):
        assert x.shape.kind == y.shape.kind == "box"
        Rx, Ry = np.asarray(x.T)[:3, :3], np.asarray(y.T)[:3, :3]
        # a box is the same box under axis permutations / sign flips: compare corners
        assert np.allclose(sorted(map(tuple, np.round(x.corners(), 6))), sorted(map(tuple, np.round(y.corners(), 6))),
                           atol=tol), (x.name, y.name)
        assert Rx.shape == Ry.shape


VARIANTS = {
    "metres_z_up": {},
    "centimetres_y_up": {"mpu": 0.01, "up": "Y"},
    "nested_scaled_turned": {"wrap": se3.make_T(se3.rot_z(0.7) @ np.diag([2.0, 0.5, 1.5]), (0.3, -0.2, 0.1))},
    "physics_colliders_and_a_visual_decoy": {"physics": True, "decoy": True, "mpu": 0.001},
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_jig_from_usd_is_the_urdf_jig(tmp_path, variant):
    """The 13 jig boxes from USD, placed at all 37 slots of the strict layout, are the
    URDF jig's 481 boxes (which tests/test_tm_tray_layout.py checks against the original)."""
    usd = write_usd(tmp_path / "jig.usda", _parts(JIG), **VARIANTS[variant])
    doc = load_document(TM / "layouts" / "strict.yaml")
    w_urdf = build_world(doc)
    doc["placements"]["jig"] = {"model": str(usd), "prim": "/part", "at": doc["placements"]["jig"]["at"]}
    w_usd = build_world(doc)
    jig = lambda w: [s for s in w.solids(objects=False) if s.entity.startswith("jig/")]  # noqa: E731
    assert len(jig(w_urdf)) == 481
    _same_solids(jig(w_urdf), jig(w_usd))
    # names: the prim path below the part prim
    assert any(s.name.endswith("/geo/Cube_01") for s in jig(w_usd))


def test_tray_object_from_usd_with_its_mass(tmp_path):
    usd = write_usd(tmp_path / "tray.usdc", _parts(TRAY), mpu=0.01, up="Y", mass=1200.0, kpu=0.001)
    doc = load_document(TM / "layouts" / "strict.yaml")
    w_urdf = build_world(doc)
    # a variant switches the model as an `extends:` file would: urdf set to null, model given
    doc["objects"]["tray"] = {**doc["objects"]["tray"], "urdf": None, "model": str(usd), "prim": "/part"}
    w_usd = build_world(doc)
    a, b = w_urdf.objects[0], w_usd.objects[0]
    assert b.mass == pytest.approx(1.2) and a.mass == pytest.approx(1.2)   # 1200 g in a kg-per-unit 0.001 stage
    assert np.allclose(a.T, b.T)
    _same_solids(a.solids, b.solids)
    # the layout's mass overrides the model's
    doc["objects"]["tray"]["mass"] = 2.0
    assert build_world(doc).objects[0].mass == 2.0


def test_instanced_parts_are_read_through_their_prototype(tmp_path):
    """Two instanceable references to one prototype: both copies collide."""
    parts = _parts(TRAY)
    proto = write_usd(tmp_path / "proto.usda", parts, root="/tray")
    stage = Usd.Stage.CreateNew(str(tmp_path / "pair.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for name, x in (("a", 0.0), ("b", 0.5)):
        p = stage.DefinePrim(f"/pair/{name}", "Xform")
        p.GetReferences().AddReference(str(proto), "/tray")
        p.SetInstanceable(True)
        _xform(p, se3.make_T(None, (x, 0, 0)))
    stage.GetRootLayer().Save()
    m = UM.load(tmp_path / "pair.usda", "/pair")
    assert len(m.solids) == 2 * len(parts)
    xs = sorted({round(float(np.asarray(s.T)[0, 3]), 6) for s in m.solids})
    one = UM.load(proto, "/tray")
    assert xs == sorted({round(float(np.asarray(s.T)[0, 3]) + dx, 6) for s in one.solids for dx in (0.0, 0.5)})


def test_collision_rules(tmp_path):
    parts = _parts(TRAY)
    usd = write_usd(tmp_path / "t.usda", parts, decoy=True)
    assert len(UM.load(usd, "/part").solids) == len(parts) + 1             # no physics marks: everything
    assert UM.load(usd, "/part").rule == "all geometry"
    sel = UM.load(usd, "/part", collision=["/part/geo/*"])                # explicit globs
    assert len(sel.solids) == len(parts)
    with pytest.raises(UM.UsdError, match="selects no geometric prim"):
        UM.load(usd, "/part", collision="/nothing/*")
    phys = write_usd(tmp_path / "p.usda", parts, physics=True, decoy=True)
    m = UM.load(phys, "/part")
    assert len(m.solids) == len(parts) and m.rule == "UsdPhysics.CollisionAPI"
    # guide purpose never collides; proxy purpose stands for render
    stage = Usd.Stage.Open(str(usd))
    UsdGeom.Imageable(stage.GetPrimAtPath("/part/visual_only")).CreatePurposeAttr(UsdGeom.Tokens.guide)
    stage.Save()
    assert len(UM.load(usd, "/part").solids) == len(parts)
    stage = Usd.Stage.Open(str(usd))
    UsdGeom.Imageable(stage.GetPrimAtPath("/part/visual_only")).CreatePurposeAttr(UsdGeom.Tokens.proxy)
    UsdGeom.Imageable(stage.GetPrimAtPath(f"/part/geo/{parts[0][0]}")).CreatePurposeAttr(UsdGeom.Tokens.render)
    stage.Save()
    m = UM.load(usd, "/part")
    assert len(m.solids) == len(parts) and "proxy" in m.rule


def test_primitives_and_mirrors(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "prims.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    c = UsdGeom.Cube.Define(stage, "/p/cube")
    c.CreateSizeAttr(0.1)
    _xform(c, se3.make_T(se3.rot_z(0.3), (1, 0, 0)) @ np.diag([1.0, 2.0, 3.0, 1.0]))
    s = UsdGeom.Sphere.Define(stage, "/p/ball")
    s.CreateRadiusAttr(0.05)
    _xform(s, np.diag([2.0, 2.0, 2.0, 1.0]))
    e = UsdGeom.Sphere.Define(stage, "/p/egg")
    e.CreateRadiusAttr(0.05)
    _xform(e, np.diag([1.0, 1.0, 2.0, 1.0]))
    y = UsdGeom.Cylinder.Define(stage, "/p/rod")
    y.CreateRadiusAttr(0.02)
    y.CreateHeightAttr(0.4)
    y.CreateAxisAttr("X")
    UsdGeom.Capsule.Define(stage, "/p/pill").CreateRadiusAttr(0.03)
    UsdGeom.Cone.Define(stage, "/p/cone").CreateRadiusAttr(0.03)
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    v = np.asarray(box.vertices) + (0.2, 0, 0)                           # off-centre: a 'mesh', not at origin
    m = _mesh(stage, "/p/mirrored", np.r_[v, [[0.25, 0.1, 0.0]]], np.asarray(box.faces))   # 9 points: not a box
    _xform(m, np.diag([-1.0, 1.0, 1.0, 1.0]))                           # a mirror
    stage.GetRootLayer().Save()
    got = {x.name: x for x in UM.load(tmp_path / "prims.usda", "/p").solids}
    cube = got["cube"]
    assert cube.shape.kind == "box" and np.allclose(sorted(cube.shape.size), [0.1, 0.2, 0.3])
    assert np.allclose(np.asarray(cube.T)[:3, 3], (1, 0, 0))
    assert got["ball"].shape.kind == "sphere" and got["ball"].shape.radius == pytest.approx(0.1)
    assert got["egg"].shape.kind == "mesh"                             # non-uniform: triangulated
    rod = got["rod"]
    assert rod.shape.kind == "cylinder" and rod.shape.length == pytest.approx(0.4)
    assert np.allclose(np.abs(np.asarray(rod.T)[:3, 2]), (1, 0, 0))      # its axis along x
    assert got["pill"].shape.kind == got["cone"].shape.kind == "mesh"
    mir = got["mirrored"]
    tm = trimesh.load(mir.shape.path, force="mesh")
    assert tm.volume > 0                                                  # faces still point outward
    centre = (np.asarray(mir.T) @ np.r_[tm.bounds.mean(0), 1.0])[:3]
    assert centre[0] == pytest.approx(-0.2, abs=1e-6)                     # mirrored to -x


def test_frames_become_features(tmp_path):
    usd = write_usd(tmp_path / "rack.usda", _parts(TRAY), root="/rack")
    stage = Usd.Stage.Open(str(usd))
    for i, x in enumerate((0.1, 0.3)):
        f = UsdGeom.Xform.Define(stage, f"/rack/slots/s{i}")
        _xform(f, se3.make_T(se3.rot_z(0.5), (x, 0, 0.2)))
        f.GetPrim().SetCustomDataByKey("rsf:role", "source")
        f.GetPrim().SetCustomData({"rsf": {"role": "source", "column": i}})
    stage.Save()
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    doc["entities"]["rack"] = {"model": str(usd), "prim": "/rack", "pose": {"xyz": [0.5, 0.0, 0.0]},
                               "frames": {"prims": "/rack/slots/*", "kind": "shelf_slot"}}
    w = build_world(doc)
    fs = w.features({"kind": "shelf_slot", "role": "source"})
    assert [f.key for f in fs] == ["rack/s0", "rack/s1"] and fs[1].get("column") == 1
    assert np.allclose(np.asarray(fs[1].T)[:3, 3], (0.8, 0.0, 0.2))
    assert w.entity("rack").solids[0].name.startswith("rack/")


def test_layout_errors(tmp_path):
    usd = write_usd(tmp_path / "t.usda", _parts(TRAY))
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    spec = {"model": str(usd), "pose": {"xyz": [0, 0, 0]}}
    for bad, match in [({**spec, "prim": "/nope"}, "top-level prims"),
                       ({**spec, "colour": "red"}, "unknown key"),
                       ({**spec, "urdf": str(TRAY)}, "exactly one of"),
                       ({"model": str(TRAY), "prim": "/x"}, "USD models only"),
                       ({"model": str(tmp_path / "t.stl")}, "a model is")]:
        d = dict(doc, entities={**doc["entities"], "thing": bad})
        with pytest.raises(LayoutError, match=match):
            build_world(d)
    with pytest.raises(LayoutError, match="unknown key"):                 # objects have no frames
        build_world(dict(doc, objects={"block": {**doc["objects"]["block"], "frames": {"prims": "*"}}}))


def test_audit_warns_about_concave_meshes(tmp_path):
    """A cup (a pocket): MuJoCo and PhysX would fill it in; a ball is fine."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "u.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    ball = trimesh.creation.icosphere(subdivisions=2, radius=0.05)       # convex, not a box
    _mesh(stage, "/p/ball", ball.vertices, ball.faces)
    cup = _cup()
    assert cup.is_watertight and cup.volume > 0
    _mesh(stage, "/p/cup", cup.vertices, cup.faces)
    stage.GetRootLayer().Save()
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    doc["entities"]["parts"] = {"model": str(tmp_path / "u.usda"), "prim": "/p", "pose": {"xyz": [1.5, 0, 0]}}
    checks = [c for c in AU.audit(build_world(doc)) if c.name == "rsf.convex_meshes"]
    assert len(checks) == 1 and checks[0].status == AU.WARN and "parts/cup" in checks[0].detail
    assert checks[0].value > 0.5                                         # the pocket is most of the hull
    doc["entities"]["parts"]["collision"] = ["/p/ball"]
    checks = [c for c in AU.audit(build_world(doc)) if c.name == "rsf.convex_meshes"]
    assert [c.status for c in checks] == [AU.PASS]


def _cup():
    """A watertight box with a square pocket from the top (concave), built face by face."""
    a, b, h, d = 0.1, 0.08, 0.05, -0.03          # outer / inner half width, half height, pocket floor z
    V = np.array([[-a, -a, -h], [a, -a, -h], [a, a, -h], [-a, a, -h],      # 0-3 bottom
                  [-a, -a, h], [a, -a, h], [a, a, h], [-a, a, h],          # 4-7 rim, outer
                  [-b, -b, h], [b, -b, h], [b, b, h], [-b, b, h],          # 8-11 rim, inner
                  [-b, -b, d], [b, -b, d], [b, b, d], [-b, b, d]])         # 12-15 pocket floor
    quads = [(0, 3, 2, 1),                                                 # bottom
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),       # outer walls
             (4, 5, 9, 8), (5, 6, 10, 9), (6, 7, 11, 10), (7, 4, 8, 11),   # rim
             (8, 9, 13, 12), (9, 10, 14, 13), (10, 11, 15, 14), (11, 8, 12, 15),   # pocket walls
             (12, 13, 14, 15)]                                             # pocket floor
    F = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    m = trimesh.Trimesh(V, F, process=False)
    trimesh.repair.fix_normals(m)
    return m


def test_cuRobo_job_scenes_travel(tmp_path):
    """Repository meshes go by relative path; cache meshes are embedded and written back."""
    import importlib.util

    from rsf.planning.plan import portable_scene
    spec = importlib.util.spec_from_file_location("cfp", ROOT / "tools" / "kaggle" / "curobo_free_plan.py")
    cfp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfp)
    cache_mesh = UM.write_obj(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
                              np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    repo_mesh = ROOT / "cells" / "tm_tray" / "assets" / "tm5s" / "meshes" / "tm5s-a1_c.stl"
    scene = {"mesh": {"a": {"file_path": str(cache_mesh), "pose": [0] * 7},
                      "b": {"file_path": str(repo_mesh), "pose": [0] * 7, "scale": [1, 1, 1]}},
             "cuboid": {"c": {"dims": [1, 1, 1], "pose": [0] * 7}}}
    p = portable_scene(scene, ROOT)
    assert "file_path" not in p["mesh"]["a"] and len(p["mesh"]["a"]["vertices"]) == 4
    assert p["mesh"]["b"]["file_path"] == "cells/tm_tray/assets/tm5s/meshes/tm5s-a1_c.stl"
    back = cfp.materialize(p, tmp_path / "work")
    assert Path(back["mesh"]["b"]["file_path"]) == repo_mesh
    got = trimesh.load(back["mesh"]["a"]["file_path"], force="mesh", process=False)
    assert np.allclose(got.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert back["cuboid"] == scene["cuboid"]


def test_a_stage_without_units_is_in_centimetres(tmp_path):
    """USD's fallback metersPerUnit is 0.01: a stage that does not say is read as cm."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "nounits.usda"))
    UsdGeom.Cube.Define(stage, "/p/c").CreateSizeAttr(10.0)
    stage.GetRootLayer().Save()
    m = UM.load(tmp_path / "nounits.usda", "/p")
    assert m.meters_per_unit == pytest.approx(0.01) and m.up_axis == "Y"   # USD's fallbacks: cm, Y up
    assert np.allclose(m.solids[0].shape.size, [0.1, 0.1, 0.1])


def test_usd_meshes_reach_the_backends(tmp_path):
    """A non-box USD mesh becomes a cached OBJ that coal and MuJoCo load."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "ball.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    ball = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    _mesh(stage, "/ball/shell", ball.vertices, ball.faces)
    stage.GetRootLayer().Save()
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    doc["entities"]["ball"] = {"model": str(tmp_path / "ball.usda"), "prim": "/ball", "pose": {"xyz": [0.6, 0, 0.5]}}
    w = build_world(doc)
    (s,) = w.entity("ball").solids
    assert s.shape.kind == "mesh" and Path(s.shape.path).parent == tmp_path / "cache" / "meshes"
    pytest.importorskip("coal")
    from rsf.collision.coal_backend import CoalChecker
    from rsf.collision import ContactPolicy
    ck = CoalChecker(w.robot.urdf, w.robot.base, ContactPolicy(0.0))
    d = ck.distance_to({}, [s])
    assert 0.0 < d < 1.0
    mujoco = pytest.importorskip("mujoco")
    from rsf.compile.mjcf import world_spec
    m = world_spec(w, robot=False).compile()
    g = m.geom("ball/shell")
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_MESH


# ── looks: the original prims referenced in the Isaac stage ─────────────────
def _fancy_part(path) -> Path:
    """A part as a CAD / Isaac export comes: centimetres, Y up, a material on every mesh,
    colliders marked, a rigid body and a joint authored, one piece instanced."""
    parts = _parts(TRAY)
    usd = write_usd(path, parts, mpu=0.01, up="Y", physics=True, root="/fixture")
    stage = Usd.Stage.Open(str(usd))
    from pxr import UsdShade
    mat = UsdShade.Material.Define(stage, "/fixture/Looks/orange")
    sh = UsdShade.Shader.Define(stage, "/fixture/Looks/orange/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.5, 0.1))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    for name, *_ in parts:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/fixture/geo/{name}")).Bind(mat)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/fixture"))
    UsdPhysics.RevoluteJoint.Define(stage, "/fixture/hinge")
    # an instanced prop from another file (its collider and rigid body inside the prototype)
    prop = Usd.Stage.CreateNew(str(Path(path).with_name("prop.usda")))
    UsdGeom.SetStageMetersPerUnit(prop, 0.01)
    UsdGeom.SetStageUpAxis(prop, UsdGeom.Tokens.y)
    c = UsdGeom.Cube.Define(prop, "/prop/knob")
    c.CreateSizeAttr(4.0)
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(prop.GetPrimAtPath("/prop"))
    prop.GetRootLayer().Save()
    inst = stage.DefinePrim("/fixture/geo/prop", "Xform")
    inst.GetReferences().AddReference(str(Path(path).with_name("prop.usda")), "/prop")
    inst.SetInstanceable(True)
    _xform(inst, se3.make_T(None, (0.0, 20.0, 0.0)))                    # 20 cm up (Y)
    stage.Save()
    return usd


def _bounds(stage, prim_path) -> np.ndarray:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return np.array([list(r.GetMin()), list(r.GetMax())])


def _solid_bounds(solids) -> np.ndarray:
    P = np.vstack([s.corners() for s in solids])
    return np.array([P.min(0), P.max(0)])


def _physics_under(stage, path) -> list:
    return [(str(p.GetPath()), a) for p in Usd.PrimRange(stage.GetPrimAtPath(path))
            for a in p.GetAppliedSchemas() if a.startswith(("Physics", "Physx"))]


def test_isaac_stage_shows_parts_by_their_original_looks(tmp_path):
    from pxr import UsdShade

    from rsf.compile.usd import world_stage
    usd = _fancy_part(tmp_path / "fixture.usda")
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    pose = {"xyz": [0.9, 0.2, 0.1], "rpy": [0, 0, 0.6]}
    doc["entities"]["fixture"] = {"model": str(usd), "prim": "/fixture", "pose": pose}
    w = build_world(doc)
    st = world_stage(w)

    # the looks sit exactly where the colliders are (cm and Y up converted, pose applied)
    fixture = w.entity("fixture")
    assert np.allclose(_bounds(st, "/World/visuals/fixture"), _solid_bounds(fixture.solids), atol=1e-5)
    # ... with their material
    meshes = [p for p in Usd.PrimRange(st.GetPrimAtPath("/World/visuals/fixture")) if p.GetTypeName() == "Mesh"]
    assert len(meshes) == len(fixture.solids) - 1                           # + the prop's Cube
    bound = {str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath()) for p in meshes}
    assert bound == {"/World/visuals/fixture/units/fixture/Looks/orange"}
    # ... and no physics: no physics schema, the joint inactive
    assert _physics_under(st, "/World/visuals") == []
    assert not st.GetPrimAtPath("/World/visuals/fixture/units/fixture/hinge").IsActive()
    # the colliders stay, invisible; the same colliders as without looks
    cols = [p for p in Usd.PrimRange(st.GetPrimAtPath("/World/static")) if p.GetCustomDataByKey("rsf:entity") == "fixture"]
    assert cols and all(UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible for p in cols)
    plain = world_stage(w, visuals=False)
    colliders = lambda s: sorted(str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))  # noqa: E731
    assert colliders(st) == colliders(plain)
    assert not plain.GetPrimAtPath("/World/visuals")
    assert all(UsdGeom.Imageable(plain.GetPrimAtPath(str(p.GetPath()))).ComputeVisibility() != UsdGeom.Tokens.invisible
               for p in cols)                                              # without looks the colliders show
    # the instanced prop: read through its prototype for the colliders, made editable in the looks
    assert any(s.name.endswith("geo/prop/knob") for s in fixture.solids)
    knob = st.GetPrimAtPath("/World/visuals/fixture/units/fixture/geo/prop/knob")
    assert knob and not knob.IsInstanceProxy() and _physics_under(st, "/World/visuals/fixture/units/fixture/geo/prop") == []

    # objects: the block's looks ride on its body
    o = w.objects[0]
    body = f"/World/objects/{o.name.replace('/', '_')}"
    assert st.GetPrimAtPath(f"{body}/visual")
    assert np.allclose(_bounds(st, f"{body}/visual"), _solid_bounds(o.world_solids()), atol=1e-6)
    assert _physics_under(st, f"{body}/visual") == []
    assert st.GetPrimAtPath(body).HasAPI(UsdPhysics.RigidBodyAPI)          # the body itself keeps its physics


def test_visual_options(tmp_path):
    usd = _fancy_part(tmp_path / "fixture.usda")
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    doc["entities"]["fixture"] = {"model": str(usd), "prim": "/fixture", "visual": False}
    assert "visual" not in build_world(doc).entity("fixture").meta
    # a URDF part with looks from a USD file
    doc["entities"]["fixture"] = {"model": str(TRAY), "visual": {"model": str(usd), "prim": "/fixture"}}
    v = build_world(doc).entity("fixture").meta["visual"]
    assert v["prims"] == ["/fixture"] and not v["keep_ops"]
    for bad in (True and {"model": str(TRAY)}, {"prim": "/x"}, "yes"):
        doc["entities"]["fixture"] = {"model": str(TRAY), "visual": bad}
        with pytest.raises(LayoutError, match="visual"):
            build_world(doc)
    # displace moves the looks with the part
    doc["entities"]["fixture"] = {"model": str(usd), "prim": "/fixture", "displace": {"xyz": [0.1, 0, 0]}}
    e = build_world(doc).entity("fixture")
    assert np.allclose(np.asarray(e.meta["visual"]["T"])[:3, 3], (0.1, 0, 0))


def test_looks_of_a_whole_stage_keep_their_placement(tmp_path):
    """No ``prim``: the part is the stage, its top-level prims stay where the stage puts them."""
    from rsf.compile.usd import world_stage
    usd = write_usd(tmp_path / "w.usda", _parts(TRAY), mpu=0.01, up="Y", root="/somewhere",
                    wrap=se3.make_T(se3.rot_z(0.4), (5.0, 1.0, 0.0)))
    stage = Usd.Stage.Open(str(usd))
    _xform(stage.GetPrimAtPath("/somewhere"), se3.make_T(se3.rot_x(0.3), (20.0, 0.0, 0.0)))
    stage.Save()
    doc = load_document(EXAMPLE / "layouts" / "main.yaml")
    doc["entities"]["thing"] = {"model": str(usd), "pose": {"xyz": [0.0, 1.0, 0.0]}}
    w = build_world(doc)
    st = world_stage(w)
    assert np.allclose(_bounds(st, "/World/visuals/thing"), _solid_bounds(w.entity("thing").solids), atol=1e-5)

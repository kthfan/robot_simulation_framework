"""World -> MuJoCo: the compiled model is the planned cell."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rsf.compile.mjcf import world_spec  # noqa: E402
from rsf.core import se3  # noqa: E402
from rsf.model.layout import load_world  # noqa: E402
from rsf.model.urdf import UrdfModel  # noqa: E402

LAYOUT = Path(__file__).resolve().parents[1] / "cells" / "tm_tray" / "layouts" / "strict.yaml"


@pytest.fixture(scope="module")
def compiled():
    w = load_world(LAYOUT)
    m = world_spec(w).compile()
    return w, m


def test_counts(compiled):
    w, m = compiled
    n_static = len(w.solids(objects=False))
    n_obj = sum(len(o.solids) for o in w.objects)
    robot_geoms = sum(1 for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith("robot/"))
    assert m.ngeom == n_static + n_obj + robot_geoms
    assert m.nu == 8 and sum(m.body(i).name.startswith("tray/") for i in range(m.nbody)) == 37


def test_robot_kinematics_match(compiled):
    w, m = compiled
    d = mujoco.MjData(m)
    model = UrdfModel.from_file(w.robot.urdf)
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(-2, 2, 6)
        for i, v in enumerate(q):
            d.qpos[m.joint(f"robot/joint_{i + 1}").qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        b = m.body("robot/flange").id
        T = se3.make_T(d.xmat[b].reshape(3, 3), d.xpos[b])
        ref = w.robot.base @ model.fk({f"joint_{i + 1}": v for i, v in enumerate(q)})["flange"]
        assert max(se3.pose_error(T, ref)) < 1e-9


def test_static_geometry_and_tray_mass(compiled):
    w, m = compiled
    s = w.solids(objects=False)[5]
    g = m.geom(s.name)
    assert np.allclose(m.geom_size[g.id], s.size / 2) and np.allclose(m.geom_pos[g.id], s.T[:3, 3])
    assert m.body_mass[m.body("tray/incoming/L0/C0").id] == pytest.approx(1.2)


def test_trays_rest_on_their_jigs(compiled):
    w, m = compiled
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ids = [m.body(o.name).id for o in w.objects]
    p0 = d.xpos[ids].copy()
    for _ in range(250):
        mujoco.mj_step(m, d)
    assert np.abs(d.xpos[ids] - p0).max() < 1e-3

"""The assembled TM tray cell assets against the measurements they must reproduce.

The expected values are the user's measurements recorded in AI-Robotics-V2
decisions D28-D30 (real gripper, D455 case, jaw stroke), and that project's
cell_core constants for the jig. They are external facts, not values derived
from the files under test (D22).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rsf.core import se3
from rsf.model.urdf import UrdfModel

ASSETS = Path(__file__).resolve().parents[1] / "cells" / "tm_tray" / "assets"
FULL = ASSETS / "tm5s_egu50.urdf"
ARM = [f"joint_{i}" for i in range(1, 7)]
JAWS = ["right_gripper_joint", "left_gripper_joint"]


def obj_vertices(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".obj":
        return np.array([[float(v) for v in ln.split()[1:4]]
                         for ln in path.read_text().splitlines() if ln.startswith("v ")])
    import trimesh
    return np.asarray(trimesh.load(path, force="mesh").vertices)


def extent_in(model: UrdfModel, geo_name: str, frame: str, q=None):
    """``(min, max)`` of a named collision geometry, in *frame*."""
    fk = model.fk(q, base=frame)
    for link in model.links.values():
        for g in link.collisions:
            if g.name == geo_name:
                V = obj_vertices(g.path)
                W = (fk[link.name] @ g.origin @ np.c_[V, np.ones(len(V))].T).T[:, :3]
                return W.min(0), W.max(0)
    raise KeyError(geo_name)


@pytest.fixture(scope="module")
def full() -> UrdfModel:
    return UrdfModel.from_file(FULL)


def test_every_mesh_resolves(full):
    for path in sorted(ASSETS.glob("*.urdf")) + sorted(ASSETS.glob("*/*.urdf")):
        m = UrdfModel.from_file(path)
        for link in m.links.values():
            for g in link.visuals + link.collisions:
                if g.kind == "mesh":
                    assert g.path is not None and g.path.is_file(), f"{path.name}: {g.filename}"


def test_joint_order_and_limits(full):
    assert full.movable_joints == ARM + JAWS or full.movable_joints == ARM + JAWS[::-1]
    lo, hi = full.limits(["joint_3"])
    assert hi[0] == pytest.approx(np.radians(158), abs=1e-6)  # tm_description: +-158 deg
    # D30: 0-0.051 m travel per jaw, 173 mm fully open.
    jlo, jhi = full.limits(JAWS)
    assert np.allclose(jhi - jlo, 0.051, atol=1e-6)


def test_calibrated_chain_matches_published_tm5s_v4():
    """findings/08 lists tm5s_v4's calibrated joint_6 origin; the arm must carry it."""
    arm = UrdfModel.from_file(ASSETS / "tm5s" / "tm5s.urdf")
    j6 = arm.joints["joint_6"].origin
    assert np.allclose(j6[:3, 3], [0.00011638, -0.13494993, 0.00006163], atol=1e-7)
    assert np.allclose(se3.matrix_to_rpy(j6[:3, :3]), [1.57033882, 2e-7, 0.00086241], atol=1e-7)


def test_gripper_body_and_plate_are_the_real_gripper(full):
    """D28/D29: body forward 0-93 mm, sideways +-61 mm; plate forward 0-321 mm,
    vertical -46..-36 mm (flange frame: x lateral, y up, z approach)."""
    lo, hi = extent_in(full, "body", "flange")
    assert hi[2] == pytest.approx(0.093, abs=1e-4)
    assert lo[0] == pytest.approx(-0.061, abs=1e-4) and hi[0] == pytest.approx(0.061, abs=1e-4)
    lo, hi = extent_in(full, "plate", "flange")
    assert lo[2] == pytest.approx(0.0, abs=1e-4) and hi[2] == pytest.approx(0.321, abs=1e-4)
    assert lo[1] == pytest.approx(-0.046, abs=1e-4) and hi[1] == pytest.approx(-0.036, abs=1e-4)
    assert hi[0] == pytest.approx(0.030, abs=1e-4)


def test_d455_case(full):
    """D30: flange x +-0.0619, y 0.0993..0.1283, z 0.0807..0.1067."""
    lo, hi = extent_in(full, "d455", "flange")
    assert np.allclose(lo, [-0.0619, 0.0993, 0.0807], atol=1e-4)
    assert np.allclose(hi, [0.0619, 0.1283, 0.1067], atol=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.0413, 0.057675])
def test_jaw_inner_faces(full, q):
    """cell_core: the gap between the jaws is 2 (0.02565 + q) (D30)."""
    qd = {j: q for j in JAWS}
    _, r_hi = extent_in(full, "right_gripper_0", "flange", qd)
    r_lo, _ = extent_in(full, "right_gripper_0", "flange", qd)
    l_lo, l_hi = extent_in(full, "left_gripper_0", "flange", qd)
    gap = r_lo[0] - l_hi[0]
    assert gap == pytest.approx(2 * (0.02565 + q), abs=5e-4)
    assert r_lo[0] == pytest.approx(-l_hi[0], abs=5e-4)  # symmetric


def test_jaws_and_plate_have_no_stray_mount_offset(full):
    """D29: tm5s.usd mounted the gripper at (0, +3.41, -1.76) mm. The jaws come from
    the v13 export but hang on the corrected egu50 joints; their top must be where
    cell_core measured it (JAW_FY_TOP 0.03956), not 3.4 mm higher."""
    _, hi = extent_in(full, "right_gripper_1", "flange", {j: 0.0 for j in JAWS})
    assert hi[1] == pytest.approx(0.0396, abs=1e-3)


def test_jig_boxes_match_cell_core():
    """cell_core.assets.JIG_BOXES (copied from tray_jig_with_label.usd)."""
    jig = UrdfModel.from_file(ASSETS / "tray_jig_with_label" / "tray_jig_with_label.urdf")
    V = np.vstack([obj_vertices(g.path) for g in jig.links["jig"].collisions])
    assert np.allclose(V.min(0), [-0.078, -0.135, -0.030], atol=1e-4)
    assert np.allclose(V.max(0), [0.078, 0.1582, 0.035], atol=1e-4)
    base = obj_vertices(ASSETS / "tray_jig_with_label" / "meshes" / "Cube.obj")
    assert np.allclose(base.max(0) - base.min(0), [0.156, 0.27, 0.005], atol=1e-6)

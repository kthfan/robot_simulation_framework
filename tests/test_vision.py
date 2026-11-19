"""rsf.sim.vision: the MuJoCo camera projects like the OpenCV pinhole it is given,
and a marker shows its image readable, right way up, to a viewer facing it."""
from __future__ import annotations



import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rsf.core import se3  # noqa: E402
from rsf.sim import vision as V  # noqa: E402

W, H, FX, FY, CX, CY = 320, 240, 300.0, 310.0, 140.3, 130.7
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


def _scene():
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.headlight.ambient = [1.0, 1.0, 1.0]
    spec.worldbody.add_body(name="rig")
    return spec


def _renders():
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco/>")
        r = mujoco.Renderer(m, 8, 8)
        r.close()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _renders(), reason="no OpenGL context (set MUJOCO_GL)")


@pytest.mark.parametrize("p", [(0.0, 0.0, 1.0), (0.08, -0.05, 0.9), (-0.1, 0.06, 1.2)])
def test_camera_projects_like_opencv(p):
    spec = _scene()
    T_cv = se3.make_T(se3.rot_x(0.3), [0.0, 0.0, 0.0])
    V.add_camera(spec, "rig", "cam", T_cv, FX, FY, CX, CY, W, H)
    pw = (T_cv @ np.r_[p, 1.0])[:3]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004] * 3, pos=pw.tolist(), rgba=[1, 1, 1, 1])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    img = V.render_grey(m, d, "cam")
    ys, xs = np.nonzero(img > 128)
    uv = V.project(T_cv, K, [pw])[0]
    assert abs(xs.mean() - uv[0]) < 0.6 and abs(ys.mean() - uv[1]) < 0.6


def test_marker_reads_right_way_up():
    """A sheet black on its left half, a white bar at its top: seen so by a camera facing it."""
    spec = _scene()
    T_cv = np.eye(4)                                          # camera at the origin, looking along +z
    V.add_camera(spec, "rig", "cam", T_cv, FX, FY, (W - 1) / 2, (H - 1) / 2, W, H)
    img = np.full((60, 120), 255, np.uint8)
    img[15:, :60] = 0                                         # left half black below a white top band
    img[15:, 60:] = 128
    # the sheet 1 m ahead, facing the camera: its z toward the camera (-z of the world), its y up (-y)
    T_sheet = se3.make_T(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 1.0])
    V.add_marker(spec, "sheet", T_sheet, (0.4, 0.2), img)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = V.render_grey(m, d, "cam").astype(float)
    cx, cy = (W - 1) // 2, (H - 1) // 2
    left, right = g[cy + 10, cx - 40], g[cy + 10, cx + 40]
    top = g[cy - 25, cx]                                      # the sheet spans +-31 px
    assert left < right < top


def test_tm_tray_label_sheet_is_the_slots_grid():
    """The printed label carries tags 4*grid_id .. +3, left to right, readable."""
    agd = pytest.importorskip("aprilgrid_detector")
    pytest.importorskip("cv2")
    from rsf.model.layout import load_world
    from rsf.model import registry
    from pathlib import Path
    world = load_world(Path(__file__).resolve().parents[1] / "cells" / "tm_tray" / "layouts" / "strict.yaml")
    from rsf_plugins.tm_tray.vision import sheet_image
    labels = registry.get("vision", "tm_tray.labels")(world)
    assert len(labels) == 37 and sorted(x["grid_id"] for x in labels) == list(range(37))
    lab = next(x for x in labels if x["key"] == "outgoing/L1/C2")
    dets = agd.Detector().detect(sheet_image(lab["grid_id"]))
    ids = [d.id for d in sorted(dets, key=lambda d: float(np.mean(np.asarray(d.corners)[:, 0])))]
    assert ids == [4 * lab["grid_id"] + i for i in range(4)]
    # the board frame sits on the sheet: z into the label, i.e. away from the robot (jig -y)
    jig = world.feature(lab["key"]).T
    assert float(lab["board_T"][:3, 2] @ jig[:3, 1]) == pytest.approx(-1.0)

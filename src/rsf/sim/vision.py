"""Cameras and printed markers in the MuJoCo cell (P5/P6): what a camera on the
robot sees, to test reading markers (AprilGrid labels) before the real cell.

* :func:`add_marker`: a printed sheet (greyscale image) as a textured quad,
  visual only (no contact). Its frame: origin at the sheet centre, x to the
  right and y up as a viewer facing it reads the image, z out of the sheet
  toward that viewer.
* :func:`add_camera`: a pinhole camera fixed to a body, given as OpenCV does
  (optical frame: z forward, x right, y down; ``fx, fy, cx, cy`` in pixels,
  image ``width x height``). MuJoCo's own camera looks along -z with y up; the
  conversion is here, once.
* :func:`render_grey`: the camera's image, 8-bit grey (luma).

Rendering needs an OpenGL context: ``MUJOCO_GL=egl`` (GPU) or ``osmesa`` (CPU).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from rsf.core import se3

#: OpenCV optical frame -> MuJoCo camera frame (flip y and z).
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def add_marker(spec, name: str, T_world, size, image: np.ndarray, body=None):
    """A sheet *size* = (width, height) m showing *image* (rows top to bottom)."""
    import mujoco
    img = np.ascontiguousarray(np.asarray(image, np.uint8))
    h, w = img.shape[:2]
    rgb = np.repeat(img[:, :, None], 3, axis=2) if img.ndim == 2 else img[:, :, :3]
    tex = spec.add_texture(name=f"{name}/tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=w, height=h, nchannel=3)
    tex.data = np.flipud(rgb).astype(np.uint8).tobytes()      # MuJoCo's rows run bottom to top
    mat = spec.add_material(name=f"{name}/mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}/tex"
    mat.emission, mat.specular, mat.shininess = 0.6, 0.0, 0.0
    W, H = float(size[0]) / 2.0, float(size[1]) / 2.0
    mesh = spec.add_mesh(name=f"{name}/mesh")
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL          # flat: no volume
    mesh.uservert = [-W, -H, 0.0, W, -H, 0.0, W, H, 0.0, -W, H, 0.0]
    mesh.usertexcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    mesh.userface = [0, 1, 2, 0, 2, 3]
    mesh.userfacetexcoord = [0, 1, 2, 0, 2, 3]
    pos, quat = _pose(T_world)
    parent = body if body is not None else spec.worldbody
    g = parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}/mesh",
                        pos=pos, quat=quat, material=f"{name}/mat")
    g.contype = g.conaffinity = 0
    g.group = 2
    return g


def add_camera(spec, body_name: str, name: str, T_body_cv, fx: float, fy: float, cx: float, cy: float,
               width: int, height: int):
    """A pinhole camera on *body_name* at *T_body_cv* (the OpenCV optical frame)."""
    body = spec.body(body_name)
    T = np.asarray(T_body_cv, float).copy()
    T[:3, :3] = T[:3, :3] @ CV_TO_MJ
    pos, quat = _pose(T)
    cam = body.add_camera(name=name, pos=pos, quat=quat)
    # pixel intrinsics: sensor 1 px = 1 unit, focal and principal point in pixels
    cam.resolution = [int(width), int(height)]
    cam.sensor_size = [float(width), float(height)]
    cam.focal_pixel = [float(fx), float(fy)]
    # principal point: MuJoCo takes the sensor centre minus this offset (checked by
    # rendering points, tests/test_vision.py); OpenCV puts pixel centres on
    # integers, so the centre of a W-wide image is at (W - 1) / 2
    cam.principal_pixel = [(width - 1) / 2.0 - float(cx), (height - 1) / 2.0 - float(cy)]
    return cam


def render_grey(model, data, camera: str, renderer=None) -> np.ndarray:
    """The camera's image as 8-bit grey, at its own resolution."""
    import mujoco
    cid = model.camera(camera).id
    W, H = (int(v) for v in model.cam_resolution[cid])
    r = renderer or mujoco.Renderer(model, H, W)
    r.update_scene(data, camera=camera)
    rgb = r.render()
    if renderer is None:
        r.close()
    return np.ascontiguousarray((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round()
                                .clip(0, 255).astype(np.uint8))


def project(T_world_cv, K, points) -> np.ndarray:
    """OpenCV pinhole projection of world *points* (n, 3) through a camera at *T_world_cv*."""
    P = np.asarray(points, float)
    Tc = np.linalg.inv(np.asarray(T_world_cv, float))
    pc = (Tc[:3, :3] @ P.T).T + Tc[:3, 3]
    uv = (np.asarray(K, float) @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def _pose(T):
    T = np.asarray(T, float)
    return T[:3, 3].tolist(), se3.quat_wxyz(T[:3, :3]).tolist()

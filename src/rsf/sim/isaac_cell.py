"""Isaac Sim replay of a planned run (P5): the same cell and the same report as
:class:`rsf.sim.mujoco_cell.MujocoCell`, with NVIDIA PhysX.

The stage is :func:`rsf.compile.usd.world_stage` of the world (objects only where
the run starts them), so the planner, MuJoCo and Isaac see one cell. The servo is
the MuJoCo cell's: the arm gravity compensated (its links ignore gravity), PD in
the joint drives (position and velocity targets from :class:`~rsf.sim.common.Reference`),
the integral term as an applied joint force; jaws are force-limited position
drives. :meth:`IsaacCell.run` returns an :class:`~rsf.sim.common.ExecReport`, judged
by :mod:`rsf.sim.judge` as the MuJoCo run is.

Needs Isaac Sim (``pip install isaacsim[all,extscache]``, Python 3.12 for 6.x, an
RTX-class GPU and driver; ``OMNI_KIT_ACCEPT_EULA=YES``). Only the low-level,
long-stable interfaces are used: ``omni.physx`` to step physics (no rendering)
and read contact reports, ``omni.physics.tensors`` (numpy) for the articulation
and the objects. Nothing is imported until a cell is made, so this module
imports anywhere.

PhysX and MuJoCo differ in contact (rigid, with a contact offset, vs soft with
``solref``/``solimp``) and friction (static and dynamic, patch friction, vs one
coefficient, elliptic cones); :class:`IsaacParams` keeps the shared physics of
:class:`~rsf.sim.common.SimParams` and adds PhysX's own solver settings.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from rsf.core import se3
from rsf.sim.common import ExecReport, Reference, SimParams


@dataclass
class IsaacParams(SimParams):
    solver: str = "TGS"
    position_iterations: int = 32
    velocity_iterations: int = 4
    gpu_dynamics: bool = False         #: CPU PhysX: one cell, deterministic, no GPU pipeline needed
    contact_offset: float = 0.002      #: m: contacts generated within this distance
    min_link_mass: float = 0.01        #: kg for massless links (PhysX needs mass on every body)
    arm_max_torque: float = 1.0e4      #: N m: drive limit (MuJoCo's servo is unlimited)
    max_joint_velocity: float = 1.0e6  #: deg/s or m/s: PhysX joint speed limit (off)
    headless: bool = True
    render_updates: int = 1            #: app updates per rendered frame of the views (RTX converges over more)
    #: how views are rendered: "step" (``rep.orchestrator.step``: waits for the frame; Isaac Sim 5.0:
    #: 45 ms per 640x360), "play" (the timeline plays, app updates render; Replicator captures while
    #: the timeline plays; 9 ms, but frames come a few renders late), "update"
    #: (app updates, timeline stopped). The app loop never steps physics in any of them
    #: (``/app/player/playSimulations`` off): ``run`` steps it. tools/isaac/live_probe.py compares them.
    render_mode: str = "step"
    #: "play" hands back frames a few renders late (RTX renders asynchronously): renders owed
    #: after a camera move so the last pose shows (tools/isaac/live_probe.py measures it)
    render_settle: int = 4
    #: before a render, write the bodies' poses from PhysX to USD (``update_transformations``),
    #: as Isaac's own physics step does when it steps PhysX directly: the renderer reads them there
    render_sync: bool = True
    #: show parts loaded from USD by their original prims (materials, textures) instead of their
    #: colliders; physics is the same either way (rsf.compile.usd, "Looks")
    visuals: bool = True
    #: read cameras (rsf.sim.reads): RTX subframes per render, renders per image (the
    #: renderer converges; the last is taken)
    read_subframes: int = 8
    read_renders: int = 2
    #: keep the read cameras' render products updating between reads (instead of switching
    #: them on per read): every view render then draws them too, but nothing is reallocated
    read_always_render: bool = True


_APP = None


def app(headless: bool = True):
    """The (one) Isaac Sim application, started on first use."""
    global _APP
    if _APP is None:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaacsim import SimulationApp
        _APP = SimulationApp({"headless": headless})
    return _APP


def start_app(params: Optional["IsaacParams"] = None):
    """Start the Isaac Sim app if this Python has Isaac Sim (else nothing): its ``pxr``
    is importable from then on, so do it before loading a layout with USD models."""
    import importlib.util
    if importlib.util.find_spec("isaacsim") is None:
        return None
    return app(True if params is None else params.headless)


def start_physics(ctx, dt: float):
    """Start PhysX on the context's stage and a numpy tensor view of it, as Isaac's own
    ``SimulationContext.initialize_physics`` does: ``(physx, simulation interface, view)``."""
    import omni.physics.tensors as tensors
    import omni.physx
    physx = omni.physx.get_physx_interface()
    sim_if = omni.physx.get_physx_simulation_interface()
    stage_id = ctx.get_stage_id()
    physx.force_load_physics_from_usd()
    physx.start_simulation()
    physx.update_simulation(dt, 0.0)
    view = tensors.create_simulation_view("numpy", stage_id=stage_id)
    view.set_subspace_roots("/")
    physx.update_simulation(dt, 0.0)
    return physx, sim_if, view


def camera_pose(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """A USD camera's pose (it looks along its -z, +y up) at *eye* looking at *target*."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:                     # looking straight up or down
        r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def _T7(x) -> np.ndarray:
    """``[px, py, pz, qx, qy, qz, qw]`` -> 4x4."""
    x = np.asarray(x, float)
    return se3.make_T(se3.quat_to_matrix((x[6], x[3], x[4], x[5])), x[:3])


class IsaacCell:
    def __init__(self, world, occupancy: Optional[dict] = None, params: Optional[IsaacParams] = None,
                 flange: str = "flange", usd_path=None, reads=None):
        start_app(params)                                    # Isaac's own pxr loads with the app
        from rsf.compile import usd as U
        from rsf.model.urdf import UrdfModel
        self.world = world
        self.p = params or IsaacParams()
        objs = world.objects if occupancy is None else tuple(o for o in world.objects if o.at in occupancy)
        self.object_names = [o.name for o in objs]
        self.flange = flange
        self.usd_path = Path(usd_path) if usd_path else Path(tempfile.mkdtemp(prefix="rsf_isaac_")) / "cell.usda"
        self.usd_path.parent.mkdir(parents=True, exist_ok=True)
        stage = U.world_stage(replace(world, objects=objs), self.usd_path, params=self.p)
        self.links = U.link_map(stage)                     # link -> (body, T_body_link)
        self.urdf = world.robot.urdf if isinstance(world.robot.urdf, UrdfModel) else UrdfModel.from_file(world.robot.urdf)
        self.base = np.asarray(world.robot.base, float)
        self._names = {}                                    # prim path -> (kind, name, body)
        for prim in stage.Traverse():
            k = prim.GetCustomDataByKey("rsf:kind")
            if k:
                self._names[str(prim.GetPath())] = (k, prim.GetCustomDataByKey("rsf:name"),
                                                    prim.GetCustomDataByKey("rsf:object"))
        self.object_paths = {o.name: f"{U.ROOT}/objects/{U.prim_name(o.name)}" for o in objs}
        self.robot_path = f"{U.ROOT}/robot"
        self._prim_name = U.prim_name
        self._started = False
        self._views, self._annot = {}, {}            # rendered views (add_view / render)
        self._cam_ops = {}                            # view -> (USD camera, its transform op)
        self._k = 0                                   # physics step (a render per step at most)
        self._rendered_k = None
        self.reads = reads                            # rsf.sim.reads: markers and read cameras (RTX)
        self._read_cams = {}                          # name -> (render product, annotator, crop, intrinsics)

    @staticmethod
    def p_headless(params) -> bool:
        return True if params is None else params.headless

    # ── Isaac side ─────────────────────────────────────────────────────────
    def _start(self):
        app(self.p.headless)
        import omni.usd
        ctx = omni.usd.get_context()
        ctx.open_stage(str(self.usd_path))
        import carb.settings
        carb.settings.get_settings().set_bool("/app/player/playSimulations", False)   # run() steps physics
        _APP.update()
        for v in self._views.values():                # cameras and lights before physics starts
            self._make_view(v)
        if self.reads is not None:
            self._make_reads()
        if self._views or self.reads is not None:
            self._timeline_for_render()
        self.physx, self.sim_if, self.view = start_physics(ctx, self.p.timestep)
        self.art = self.view.create_articulation_view(self.robot_path)
        meta = self.art.shared_metatype
        self.dof_names = list(meta.dof_names)
        self.link_names = list(meta.link_names)
        self.objs = {n: self.view.create_rigid_body_view(p) for n, p in self.object_paths.items()}
        self._obj0 = {n: v.get_transforms().copy() for n, v in self.objs.items()}     # where the run starts them
        self._idx = np.arange(1, dtype=np.int32)
        self._started = True

    # ── views: RTX pictures of the run ─────────────────────────────────────
    def add_view(self, view) -> None:
        """A camera :meth:`render` can draw (:class:`rsf.sim.common.View`)."""
        self._views[view.name] = view
        if self._started:
            self._make_view(view)

    def _make_view(self, v) -> None:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
        st = omni.usd.get_context().get_stage()
        root = f"{self.robot_path.rsplit('/', 1)[0]}/rsf_live"
        if not st.GetPrimAtPath(f"{root}/dome"):      # the compiled stage has no lights
            UsdLux.DomeLight.Define(st, f"{root}/dome").CreateIntensityAttr(1000.0)
            UsdLux.DistantLight.Define(st, f"{root}/sun").CreateIntensityAttr(2500.0)
        cam = UsdGeom.Camera.Define(st, f"{root}/{self._prim_name(v.name)}")
        ha = 20.955                                    # mm, USD's default aperture
        cam.CreateHorizontalApertureAttr(ha)
        cam.CreateVerticalApertureAttr(ha * v.height / v.width)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        self._cam_ops[v.name] = (cam, xf.AddTransformOp())
        self._pose_camera(v)
        rp = rep.create.render_product(str(cam.GetPath()), (int(v.width), int(v.height)))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        self._annot[v.name] = ann

    # ── read cameras (rsf.sim.reads): RTX images of the printed markers ──────
    def _lights(self, st) -> None:
        from pxr import UsdLux
        root = f"{self.robot_path.rsplit('/', 1)[0]}/rsf_live"
        if not st.GetPrimAtPath(f"{root}/dome"):      # the compiled stage has no lights
            UsdLux.DomeLight.Define(st, f"{root}/dome").CreateIntensityAttr(1000.0)
            UsdLux.DistantLight.Define(st, f"{root}/sun").CreateIntensityAttr(2500.0)

    def _body_prim(self, st, link: str):
        from pxr import Usd
        body, T_bl = self.links[link]
        name = self._prim_name(body)
        prim = st.GetPrimAtPath(f"{self.robot_path}/{name}")
        if not prim:
            prim = next(p for p in Usd.PrimRange(st.GetPrimAtPath(self.robot_path)) if p.GetName() == name)
        return prim, np.asarray(T_bl, float)

    def _make_reads(self) -> None:
        """Every marker as a textured, self-lit quad; every read camera on its link.

        A USD camera has its principal point at the image centre, the calibrated one
        does not: the camera renders a slightly larger image centred on the optical
        axis, and :meth:`camera_image` crops the calibrated size out of it at whole
        pixels, so the image's own principal point is within half a pixel of the
        calibrated one and is handed to the detector exactly."""
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        st = omni.usd.get_context().get_stage()
        self._lights(st)
        tex_dir = self.usd_path.parent / "markers"
        tex_dir.mkdir(parents=True, exist_ok=True)
        root = f"{self.robot_path.rsplit('/', 1)[0]}/rsf_markers"
        for i, mk in enumerate(self.reads.markers):
            png = tex_dir / f"marker_{i}.png"
            Image.fromarray(np.asarray(mk.image, np.uint8)).save(png)
            base = f"{root}/m{i}"
            xf = UsdGeom.Xform.Define(st, base)
            xf.AddTransformOp().Set(Gf.Matrix4d(*np.asarray(mk.T_sheet, float).T.ravel().tolist()))
            w, h = float(mk.size[0]) / 2.0, float(mk.size[1]) / 2.0
            mesh = UsdGeom.Mesh.Define(st, f"{base}/sheet")
            mesh.CreatePointsAttr([(-w, -h, 0.0), (w, -h, 0.0), (w, h, 0.0), (-w, h, 0.0)])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
            mesh.CreateDoubleSidedAttr(False)
            st_pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                             UsdGeom.Tokens.vertex)
            st_pv.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])   # image rows top to bottom: t up
            mat = UsdShade.Material.Define(st, f"{base}/mat")
            sh = UsdShade.Shader.Define(st, f"{base}/mat/pbr")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            rd = UsdShade.Shader.Define(st, f"{base}/mat/st")
            rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(st, f"{base}/mat/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
            tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            rgb = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            # self-lit only: the printed sheet looks the same under any light. Lit and self-lit
            # together over-exposed the white, which ate into the black tags: boards read
            # 0.1-0.2 % too far (the first RTX run, 2026-11-28)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        pix = 0.003                                    # mm per pixel (any: only ratios matter)
        cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])     # USD cameras look along -z, y up
        for c in self.reads.cameras.values():
            pad_x = int(math.ceil(abs(c.width / 2.0 - c.cx))) + 1
            pad_y = int(math.ceil(abs(c.height / 2.0 - c.cy))) + 1
            Wb, Hb = c.width + 2 * pad_x, c.height + 2 * pad_y
            x0 = int(round((Wb - 1) / 2.0 - c.cx))      # OpenCV: pixel centres on integers
            y0 = int(round((Hb - 1) / 2.0 - c.cy))
            prim, T_bl = self._body_prim(st, c.link)
            cam = UsdGeom.Camera.Define(st, f"{prim.GetPath()}/rsf_read_{self._prim_name(c.name)}")
            cam.CreateFocalLengthAttr(float(c.fx) * pix)
            cam.CreateHorizontalApertureAttr(Wb * pix)
            cam.CreateVerticalApertureAttr(Hb * pix)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            T = T_bl @ np.asarray(c.T_link_cv, float) @ cv_to_usd
            cxf = UsdGeom.Xformable(cam)
            cxf.ClearXformOpOrder()
            cxf.AddTransformOp().Set(Gf.Matrix4d(*T.T.ravel().tolist()))
            rp = rep.create.render_product(str(cam.GetPath()), (Wb, Hb))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            self._enable(rp, bool(self.p.read_always_render))
            self._read_cams[c.name] = (rp, ann, (x0, y0),
                                       (float(c.fx), float(c.fx), (Wb - 1) / 2.0 - x0, (Hb - 1) / 2.0 - y0))

    @staticmethod
    def _enable(rp, on: bool) -> None:
        ht = getattr(rp, "hydra_texture", None)
        if ht is not None and hasattr(ht, "set_updates_enabled"):
            ht.set_updates_enabled(on)

    def camera_image(self, name: str):
        """The read camera *name*'s grey image of the current state (RTX) and the
        intrinsics it was rendered with ``(fx, fy, cx, cy)`` (square pixels: fy = fx)."""
        import omni.replicator.core as rep
        if not self._started:
            self._start()
        rp, ann, (x0, y0), K = self._read_cams[name]
        c = self.reads.cameras[name]
        self._timeline_for_render()
        if self.p.render_sync:
            self.sync_render()
        if not self.p.read_always_render:
            self._enable(rp, True)
        img = None
        for _ in range(max(1, int(self.p.read_renders))):
            try:
                rep.orchestrator.step(rt_subframes=int(self.p.read_subframes), delta_time=0.0, pause_timeline=True)
            except TypeError:
                rep.orchestrator.step()
            d = ann.get_data()
            img = d.get("data") if isinstance(d, dict) else d
        if not self.p.read_always_render:
            self._enable(rp, False)
        self._rendered_k = None                          # the views draw again
        img = np.asarray(img)[..., :3].astype(float)
        grey = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).round().clip(0, 255)
        return np.ascontiguousarray(grey[y0:y0 + c.height, x0:x0 + c.width].astype(np.uint8)), K

    def link_pose(self, link: str) -> np.ndarray:
        """World pose of a robot link in the current state (PhysX)."""
        body, T_bl = self.links[link]
        return _T7(self.art.get_link_transforms()[0][self.link_names.index(self._prim_name(body))]) @ T_bl

    def _pose_camera(self, v) -> None:
        from pxr import Gf
        cam, op = self._cam_ops[v.name]
        va = 20.955 * v.height / v.width
        cam.GetFocalLengthAttr().Set(va / (2.0 * math.tan(math.radians(v.fov_deg) / 2.0)))
        op.Set(Gf.Matrix4d(*camera_pose(v.eye, v.target).T.ravel().tolist()))

    def move_view(self, name: str, eye, target, fov_deg=None) -> None:
        """Move the view *name*'s camera (a USD camera: not physics); the next render draws it."""
        v = self._views[name]
        v = self._views[name] = replace(v, eye=tuple(map(float, eye)), target=tuple(map(float, target)),
                                        fov_deg=float(v.fov_deg if fov_deg is None else fov_deg))
        if name in self._cam_ops:
            self._pose_camera(v)
        self._rendered_k = None                        # draw again even if physics has not stepped

    def render(self, name: str):
        """The view *name* of the current state, ``(h, w, 3)`` uint8 RGB (``None`` while the
        renderer has nothing yet). One render (``render_mode``) draws every view; physics
        does not step in it (``/app/player/playSimulations`` is off from the start)."""
        if not self._started:
            self._start()
        if self._rendered_k != self._k:
            self._timeline_for_render()
            if self.p.render_sync:
                self.sync_render()
            mode = self.p.render_mode
            for _ in range(max(1, int(self.p.render_updates))):
                if mode == "step":
                    import omni.replicator.core as rep
                    try:
                        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
                    except TypeError:                  # older Replicator signatures
                        rep.orchestrator.step()
                else:
                    _APP.update()
            self._rendered_k = self._k
        img = self._annot[name].get_data()
        if isinstance(img, dict):                      # some Replicator versions wrap it
            img = img.get("data")
        img = np.asarray(img)
        if img.ndim != 3 or img.size == 0:
            return None
        return img[..., :3].astype(np.uint8, copy=False)

    @property
    def render_latency(self) -> int:
        """Renders a picture lags behind (0: synchronous)."""
        return int(self.p.render_settle) if self.p.render_mode == "play" else 0

    def sync_render(self) -> None:
        """PhysX's current body poses -> USD (what the renderer draws)."""
        try:
            self.physx.update_transformations(False, True, False, False)
        except TypeError:                              # older signature
            self.physx.update_transformations(False, True)

    def render_lag(self, link: Optional[str] = None) -> float:
        """How far (m) the USD pose of *link*'s body (default: the flange's) is from PhysX's:
        about 0 when the renderer can see the run (with ``render_sync``)."""
        import omni.usd
        from pxr import Usd, UsdGeom
        body, _ = self.links[link or self.flange]
        name = self._prim_name(body)
        st = omni.usd.get_context().get_stage()
        prim = st.GetPrimAtPath(f"{self.robot_path}/{name}")
        if not prim:
            prim = next(p for p in Usd.PrimRange(st.GetPrimAtPath(self.robot_path)) if p.GetName() == name)
        usd = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T[:3, 3]
        phys = np.asarray(self.art.get_link_transforms()[0][self.link_names.index(name)][:3], float)
        return float(np.linalg.norm(usd - phys))

    def _timeline_for_render(self) -> None:
        if self.p.render_mode not in ("play", "step", "update"):
            raise ValueError(f"render_mode {self.p.render_mode!r}: play, step or update")
        if self.p.render_mode == "play":
            import omni.timeline
            tl = omni.timeline.get_timeline_interface()
            if not tl.is_playing():
                tl.play()

    def _path_name(self, i: int):
        from pxr import PhysicsSchemaTools
        return str(PhysicsSchemaTools.intToSdfPath(i))

    def _who(self, path: str):
        """``(is_robot, name)`` for a collider or body path."""
        while path and path not in self._names:
            path = path.rsplit("/", 1)[0]
        kind, name, obj = self._names.get(path, ("", path, None))
        if kind.startswith("robot"):
            return True, name
        if kind == "object_solid":
            return False, obj
        return False, name

    def _contacts(self, t, out):
        headers, data = self.sim_if.get_contact_report()
        for h in headers:
            c0 = self._path_name(getattr(h, "collider0", h.actor0))
            c1 = self._path_name(getattr(h, "collider1", h.actor1))
            r0, n0 = self._who(c0)
            r1, n1 = self._who(c1)
            if r0 == r1:
                continue
            sep = min((data[h.contact_data_offset + k].separation for k in range(h.num_contact_data)), default=0.0)
            if sep >= -self.p.contact_tol:
                continue
            link, other = (n0, n1) if r0 else (n1, n0)
            out.append({"t": round(t, 4), "link": link, "other": other, "depth": float(-sep)})

    def _flange_actual(self, links_T):
        body, T_bl = self.links[self.flange]
        return _T7(links_T[self.link_names.index(self._prim_name(body))]) @ T_bl

    # ── the run ────────────────────────────────────────────────────────────
    def run(self, traj: dict, t_end: Optional[float] = None, progress=None, on_sample=None) -> ExecReport:
        """Replay a ``rsf.trajectory/1`` dict; *on_sample(t, cell)* at every sample."""
        if not self._started:
            self._start()
        p = self.p
        ref = Reference(traj)
        names, T, Q = ref.names, ref.T, ref.Q
        col = np.array([names.index(n) for n in self.dof_names])     # trajectory column of each DOF
        arm = np.array([self.urdf.joints[n].type != "prismatic" for n in self.dof_names])
        kp = np.zeros(len(self.dof_names))
        kp[arm] = np.array(p.arm_kp[:int(arm.sum())]) * p.arm_ki_per_s
        art, idx = self.art, self._idx
        for n, v in self.objs.items():                    # every run starts from the layout
            v.set_transforms(self._obj0[n], idx)
            v.set_velocities(np.zeros((1, 6), np.float32), idx)
        art.set_dof_positions(Q[0][col][None], idx)
        art.set_dof_velocities(np.zeros((1, len(col))), idx)
        art.set_dof_position_targets(Q[0][col][None], idx)
        t_stop = (T[-1] if t_end is None else min(t_end, T[-1])) + p.settle
        n_steps = int(math.ceil(t_stop / p.timestep))
        every_s = max(1, int(round(p.sample_dt / p.timestep)))
        every_c = max(1, int(round(p.contact_dt / p.timestep)))
        ts, fl, fp, qs, contacts = [], [], [], [], []
        objs = {n: [] for n in self.object_names}
        integ = np.zeros(len(col))
        tool = self.flange
        wall = time.time()
        for k in range(n_steps + 1):
            t = k * p.timestep - p.settle
            q_ref, qd_ref = ref(t)
            qr, qdr = q_ref[col], qd_ref[col]
            q = art.get_dof_positions()[0]
            e = np.where(arm, qr - q, 0.0)
            integ += e * p.timestep
            art.set_dof_position_targets(qr[None], idx)
            art.set_dof_velocity_targets(np.where(arm, qdr, 0.0)[None], idx)
            art.set_dof_actuation_forces((kp * integ)[None], idx)
            if k % every_c == 0 and t >= 0:
                self._contacts(t, contacts)
            if k % every_s == 0 and t >= 0:
                ts.append(t)
                fl.append(self._flange_actual(art.get_link_transforms()[0]))
                for n, v in self.objs.items():
                    objs[n].append(_T7(v.get_transforms()[0]))
                fp.append(self.base @ self.urdf.link_pose(tool, dict(zip(names, q_ref))))
                qa = np.array(q_ref, float)
                qa[col] = q
                qs.append(qa)
                if on_sample is not None:
                    on_sample(t, self)
                if progress and len(ts) % 500 == 0:
                    progress(t, time.time() - wall)
            self.physx.update_simulation(p.timestep, k * p.timestep)
            self._k += 1
        return ExecReport(np.array(ts), np.array(fl), np.array(fp), np.array(qs),
                          {n: np.array(v) for n, v in objs.items()}, contacts, list(traj.get("segments", [])),
                          {"backend": "isaac", "params": dict(vars(p)), "usd": str(self.usd_path),
                           "wall_s": round(time.time() - wall, 1), "steps": n_steps})

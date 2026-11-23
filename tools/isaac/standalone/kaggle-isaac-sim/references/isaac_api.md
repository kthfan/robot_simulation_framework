# Isaac Sim 6.x headless: patterns that work

Notes from driving Isaac Sim 6.2 headless on Kaggle T4s with plain Python, without
`isaacsim.core.api` wrappers. Every snippet assumes `from isaac_util import app, start_physics,
pose7_to_T, report`, which works on Kaggle because `$ISAAC_KIT` is on `PYTHONPATH`.

## Contents
1. Process lifecycle
2. Stepping physics yourself
3. Articulations (robots) through the tensor API
4. Rigid bodies
5. Contacts
6. Authoring physics USD (with usd-core, off Kaggle, or with Isaac's pxr)
7. Rendering with Replicator
8. Known noise and failures

## 1. Process lifecycle

```python
sim = app()                 # SimulationApp({"headless": True}); do this first
import omni.usd             # only now: omni.*, and Isaac's own pxr
ctx = omni.usd.get_context()
ctx.open_stage("/abs/path/scene.usda")    # or ctx.new_stage()
sim.update()
...
report(results, "reports/isaac/x.json")   # write everything first
sim.close()                                # ends the Python process: nothing after it runs
```

- Importing `pxr` before the app picks up a different `pxr`, or none.
- Pass absolute paths to `open_stage`.
- Startup takes about 20–40 s with a warm cache.

## 2. Stepping physics yourself

```python
physx, sim_if, view = start_physics(ctx, dt)        # loads the physics from the stage
for k in range(n):
    ...set targets / read state through `view`...
    physx.update_simulation(dt, k * dt)
```

- `start_physics` does `force_load_physics_from_usd()`, `start_simulation()` and one step, then
  `omni.physics.tensors.create_simulation_view("numpy", stage_id=ctx.get_stage_id())` and
  `set_subspace_roots("/")`.
- Without `stage_id`, the view fails with "Failed to get a valid attached USD stage id".
- `update_simulation` steps exactly once. No timeline or app update is needed for physics.
- The scene's `physxScene:timeStepsPerSecond` should match `1/dt`.

## 3. Articulations (robots) through the tensor API

```python
art = view.create_articulation_view("/World/robot")   # the prim with ArticulationRootAPI
names = list(art.shared_metatype.dof_names)            # DOF order of every array below
links = list(art.shared_metatype.link_names)
idx = np.arange(1, dtype=np.int32)                     # one articulation instance
art.set_dof_positions(q0[None], idx); art.set_dof_velocities(np.zeros((1, len(names))), idx)
art.set_dof_position_targets(q_ref[None], idx)         # the joint drives (PD)
art.set_dof_velocity_targets(qd_ref[None], idx)
art.set_dof_actuation_forces(tau[None], idx)           # extra joint effort, e.g. an integral term
q = art.get_dof_positions()[0]
T_link = pose7_to_T(art.get_link_transforms()[0][links.index("flange")])  # [p, q_xyzw]
```

- Angular DOFs are in radians in the tensor API.
- Drive gains on the USD joint are per degree: `stiffness = kp[N·m/rad] · π/180`, and the
  same for damping. Revolute limits are in degrees.
- A PD drive alone leaves a steady-state error under load. Either add an integral term
  through `set_dof_actuation_forces`, or disable gravity on the robot bodies
  (`physxRigidBody:disableGravity`) to model a gravity-compensated controller.
- Link names in the view are the prim names of the rigid bodies. Links joined by fixed
  joints are best merged into one body when the USD is authored.

## 4. Rigid bodies

```python
rb = view.create_rigid_body_view("/World/objects/box")    # a prim path, or a pattern "/World/objects/*"
T = pose7_to_T(rb.get_transforms()[0])
rb.set_transforms(x7[None], idx); rb.set_velocities(np.zeros((1, 6), np.float32), idx)
```

To start every episode from the same state, read the transforms once after
`start_physics` and write them back before each run.

## 5. Contacts

Apply `PhysxContactReportAPI` with `physxContactReport:threshold = 0` to the bodies you care
about. Then, after a step:

```python
from pxr import PhysicsSchemaTools
headers, data = sim_if.get_contact_report()
for h in headers:
    a = str(PhysicsSchemaTools.intToSdfPath(h.collider0)); b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
    sep = min(data[h.contact_data_offset + k].separation for k in range(h.num_contact_data))
    # sep < 0: penetration depth -sep (m). PhysX creates contacts from contactOffset away
    # with sep > 0, so filter on sep < -tolerance for real touches.
```

- The report covers only the last step. Sample it every few ms rather than every step: it is slow.
- Tag your prims with `customData` (for example `{"my:name": ...}`), so contact paths map
  back to model names without string parsing.

## 6. Authoring physics USD

- **usd-core** (pip) has `UsdPhysics` but no `PhysxSchema`. Apply PhysX schemas by name with
  `prim.AddAppliedSchema("PhysxSceneAPI")`, then
  `prim.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")`.
  Isaac reads these tokens. Author stages on CPU this way and test them locally.
- Never install usd-core into the Isaac environment: its `pxr` conflicts with Isaac's.
- Units: `UsdGeom.SetStageUpAxis(st, "Z")` and `SetStageMetersPerUnit(st, 1.0)`.
- Mind the xform op order: `AddTranslateOp()` then `AddScaleOp()` scales first and then
  translates. In the other order the translation is scaled too.
- Use `UsdGeom.Cube` with `CreateSizeAttr(1.0)` and a scale op for boxes. **A Cube's default
  size is 2** (it spans −1 to 1), so a box authored without the size is twice as big.
- Meshes: `UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")`
  for dynamic bodies. Triangle meshes only collide as static geometry.
- **Articulation**: `ArticulationRootAPI` on the robot root prim, and
  `physxArticulation:enabledSelfCollisions = False` unless you need self-collision.
  - Fix the base with a `FixedJoint` whose `body1` is the root link.
  - Joints: `RevoluteJoint` or `PrismaticJoint`, `axis "X"`, and `localPos0/1` and
    `localRot0/1` set so the joint frame's X is the joint axis.
  - Drives: `DriveAPI.Apply(joint, "angular"|"linear")`, type `force`, stiffness, damping,
    maxForce; `physxJoint:armature` for rotor inertia.
- Mass: `MassAPI` with mass, `centerOfMass`, `diagonalInertia` and `principalAxes`. Bodies
  without mass get it from the default density, which differs from MuJoCo's.
- Friction: `UsdPhysics.MaterialAPI` on a `UsdShade.Material` bound with purpose
  `"physics"`. Use `physxMaterial:frictionCombineMode "max"` to match MuJoCo's
  max-of-pair rule.
- Contact offsets: `physxCollision:contactOffset` (for example 2 mm) and
  `physxCollision:restOffset 0`.
- Solver: TGS, with `physxArticulation:solverPositionIterationCount` 16–32 for stiff grasps.
- `physxScene:enableGPUDynamics False` is fine for one robot: CPU PhysX is deterministic run to run.

## 7. Rendering with Replicator

```python
import omni.replicator.core as rep
from pxr import UsdLux
UsdLux.DomeLight.Define(st, "/dome").CreateIntensityAttr(1000.0)        # authored stages rarely have lights
cam = rep.create.camera(position=(2, -2, 1.4), look_at=(0, 0, 0.5))
rp = rep.create.render_product(cam, (1280, 960))
ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
for _ in range(3): rep.orchestrator.step()                              # RTX accumulates over frames
img = np.asarray(ann.get_data())                                        # H x W x 4 uint8
```

- Other annotators: `distance_to_camera`, `semantic_segmentation`, `bounding_box_2d_tight`
  and `camera_params`.
- Pass `width` and `height` to the app config when you render large images.
- On a T4 with a warm shader cache a frame takes 0.1–0.5 s. A cold cache spends about 10
  minutes on the first frame.
- To move the camera between frames, use `with cam: rep.modify.pose(position=..., look_at=...)`,
  or author a `UsdGeom.Camera` and use its path in `render_product`.

## 8. Known noise and failures

- Harmless log noise: audio device lines, GLFW, "failed to open the default display",
  "frictionType is deprecated", missing-icon warnings.
- `ERROR_INCOMPATIBLE_DRIVER` from Vulkan means the NVIDIA EGL vendor is not registered
  (`vulkan_setup.sh` does it). Physics works without Vulkan; rendering does not.
- `create_simulation_view` returns an invalid view: call it after `start_simulation()` and
  one step, with `stage_id`.
- A kernel that hangs on the first render with no output is compiling shaders. Give it 15
  minutes, or rebuild the cache if `ISAAC_ENV_STALE` was reported.

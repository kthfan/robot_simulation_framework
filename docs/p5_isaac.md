# P5 — Isaac Sim adapter

Date: 2026-11-22. Code: `src/rsf/compile/usd.py` (world → USD), `src/rsf/sim/isaac_cell.py`
(the run), `src/rsf/sim/common.py` (what both simulators share), `tools/sim_workflow.py --backend isaac`,
`tools/isaac/probe.py`. Tests: `tests/test_usd.py`.

The milestone (PLAN §7, P5): *the Isaac adapter loads the same world*. Same cell,
same servo, same report, same judge as the MuJoCo run, so a plan can be replayed in
either and the verdicts compared.

## Design

```
World ──rsf.compile.mjcf──► MjSpec ──► MujocoCell ─┐
   └────rsf.compile.usd───► USD stage ─► IsaacCell ─┼─► ExecReport ─► rsf.sim.judge
trajectory ─► rsf.sim.common.Reference (q_ref, qd_ref) ┘
```

| part | choice | why |
|---|---|---|
| scene format | USD with the core `UsdPhysics` schemas, PhysX settings as their attribute tokens | `usd-core` alone authors and tests it (no Isaac on CPU); Isaac's PhysX reads the same tokens |
| robot | the URDF written as an articulation here, not Isaac's URDF importer | the joints and frames are exactly the planner's; checked against the URDF's FK (< 1e-6) |
| links | links joined by fixed joints are one rigid body (named after the top link); every link is an `Xform` under its body with its collision meshes | PhysX needs mass on every body (`flange`, `gripper_base` have none); contacts still name the link (`gripper_body`, not `link_6`) |
| collision | boxes stay boxes; meshes as convex hulls (`MeshCollisionAPI convexHull`) | what MuJoCo collides, so the two runs see the same shapes |
| mass | objects: the object's mass shared by volume over its solids; links: URDF inertials, merged by the parallel axis theorem; links without `<inertial>` (the jaws) from their geometry at 1000 kg/m³ | MuJoCo's rules; masses and centres of mass checked against the compiled MuJoCo model |
| servo | arm: PD in the joint drives (`kp`, `kv` of `SimParams`; PhysX angular gains are per degree, converted), the integral as an applied joint force, links ignore gravity (= gravity compensation); jaws: position drive, 100 N limit | MuJoCo's servo, term for term |
| friction | materials `robot` 0.4, `object` 0.3, `static` 0.5, combine `max` | MuJoCo's combine rule; the plate–tray pair gets the robot value |
| contacts | `PhysxContactReportAPI` on the robot's bodies; per pair the deepest separation; robot–robot off (articulation self-collision off) | the judge's contact record: `{t, link, other, depth}` |
| runtime | `omni.physx` steps physics without rendering; `omni.physics.tensors` (numpy) for the articulation and objects | the long-stable low-level interfaces; nothing imported until a cell is made |

Names: USD prim names cannot contain `/`; every prim that stands for a model element
carries `customData` `rsf:name` (the solid / object / link / joint name), `rsf:kind`
and, for links, `rsf:body`, so contacts and poses map back to the names the judge uses.

Stage layout:

```
/World                      defaultPrim, Z up, metres, kilograms
  physicsScene              1 kHz, TGS, CPU dynamics
  materials/{robot,object,static}
  static/<solid>            colliders (504 boxes in strict)
  objects/<object>          rigid bodies, colliders below (the trays where the run starts them)
  robot                     ArticulationRootAPI, self-collision off
    base, link_1 … link_6, left_gripper, right_gripper     rigid bodies
      <link>/collision_k    convex hulls
    joints/root_fixed, joint_1 … joint_6, *_gripper_joint
```

## What differs from MuJoCo (on purpose, documented)

| | MuJoCo | PhysX |
|---|---|---|
| contact | soft (`solref` 2 ms, `solimp` 0.95–0.99) | rigid, contact offset 2 mm, rest offset 0 |
| friction | one coefficient, elliptic cones, noslip | static = dynamic = the class value, patch friction |
| solver | implicit-fast, 1 kHz | TGS, 32 position / 4 velocity iterations, 1 kHz |
| drive limit | none | 10 kN m (`arm_max_torque`) |

`IsaacParams` (a `SimParams` with PhysX's own fields) holds these; `--set` changes
them as for MuJoCo.

## Running

```bash
pip install usd-core trimesh                 # authoring and tests (CPU)
PYTHONPATH=src python -m pytest tests/test_usd.py
```

Isaac Sim itself (Python 3.12 for 6.x, an RTX-class GPU):

```bash
pip install 'isaacsim[all,extscache]==6.2.0.0' --extra-index-url https://pypi.nvidia.com
OMNI_KIT_ACCEPT_EULA=YES python tools/isaac/probe.py                      # can it run here?
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src python tools/sim_workflow.py cells/tm_tray/layouts/strict.yaml \
    cells/tm_tray/workflows/inspect_and_sort.yaml --plan cells/tm_tray/plans/strict_inspect_and_sort_gpu.json \
    --backend isaac --transfers 2 --out reports/isaac_strict_1cycle.json
```

## Status

* Compiler and its checks: done (`tests/test_usd.py`, 8 tests): FK of the authored
  articulation = URDF FK (< 1e-6 over random configurations), bodies authored where
  their joints put them, the flange frame, drives and limits, gravity off on the arm,
  static boxes and objects where the world has them, masses and centres of mass =
  MuJoCo's, names and physics materials, file round trip; the Isaac cell's name maps
  (contacts, flange) without Isaac.
* Isaac runtime: see below (Kaggle).

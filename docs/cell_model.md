# Cell model: assets, layout, workflow (P1)

A cell is three kinds of file, all under `cells/<cell>/`:

| File | Says | Format |
|---|---|---|
| `assets/` (+ `asset_spec.yaml`) | what the hardware is: robot, tool, fixtures, objects | URDF + meshes |
| `layouts/*.yaml` | where everything stands in this installation | `rsf.layout/1` |
| `workflows/*.yaml` | what the cell does, in what order | `rsf.workflow/1` |
| `plugin/<name>/` | cell-specific generators, checks (later: skills) | Python package |

`rsf.model.layout.load_world(layout)` compiles a layout into a **World**: every
solid in world coordinates, named frames (**features**, e.g. slots), movable
**objects** (trays) and the **robot** (URDF + base pose). Every backend reads the
World, so collision checkers, simulators and viewers cannot disagree about the
cell.

```bash
PYTHONPATH=src python -m rsf world  cells/tm_tray/layouts/strict.yaml
PYTHONPATH=src python -m rsf audit  cells/tm_tray/layouts/strict.yaml
PYTHONPATH=src python -m rsf expand cells/tm_tray/layouts/strict.yaml cells/tm_tray/workflows/inspect_and_sort.yaml
PYTHONPATH=src python -m rsf scene  cells/tm_tray/layouts/strict.yaml --set entities.pedestal.params.leg_height=0.6
```

## Layout (`rsf.layout/1`)

```yaml
schema: rsf.layout/1
name: strict
extends: as_built.yaml            # deep merge underneath (mappings merge, lists replace)
plugins: [../plugin/tm_tray]      # imported first; registers generators and checks
process: {standoff_near: 0.2768}  # free-form process parameters (skills, checks)
entities:                         # built in order; later entities may use earlier features
  pedestal: {generator: tm_tray.pedestal, params: {...}}
  incoming: {generator: tm_tray.rack, params: {...}}
  fence:    {urdf: ../assets/fence.urdf, pose: {xyz: [0, 1, 0], rpy: [0, 0, 0]}}
placements:                       # a fixed copy at every selected feature
  jig: {urdf: ../assets/tray_jig_with_label/tray_jig_with_label.urdf, at: {kind: slot}}
objects:                          # movable, initially at selected features
  tray: {urdf: ../assets/tray/tray.urdf, at: {kind: slot}, offset: {xyz: [0, 0, 0.035]}, initially: all}
robot: {urdf: ../assets/tm5s_egu50.urdf, base: {feature: pedestal/arm_base}, tool_frame: flange}
audit: ["tm_tray.*", "rsf.*"]     # optional: which checks run (default all)
```

- **Poses**: `{xyz, rpy}`, `{xyz, quat_wxyz}`, `{matrix}`, or `{feature: key, offset: pose}`.
- **Overrides**: dotted paths into the resolved document
  (`entities.pedestal.params.leg_height`). A path that does not exist is an
  error, so a typo cannot silently do nothing. Variants are small files that
  `extends:` a record (`strict.yaml` changes one number of `as_built.yaml`).
- **Paths** are relative to the file that wrote them, also across `extends`.
- URDF meshes that are exact axis-aligned boxes become box solids.

## Selectors

Used by layouts (`at:`) and workflows (`sets:`):
`"incoming/L0/*"` (key glob), `{kind: slot, role: outgoing, layer: [0, 3], z: {min: 1.0}}`
(all must match; list = one of; `{min,max}` = range), or a list of selectors (union).

## Workflow (`rsf.workflow/1`)

```yaml
sets: {incoming: {kind: slot, role: incoming}, outgoing: {...}, inspection: {...}}
initial: {tray: [incoming]}                  # occupancy to plan from
scheduler:
  type: rsf.ordered
  params: {bind: {source:      {set: incoming, order: [-height, column], state: occupied, on_none: request_refill},
                  destination: {set: outgoing, order: [height, column],  state: empty,    on_none: request_removal}}}
cycle:
  - {skill: transfer, args: {object: tray, from: $source, to: $inspection}}
  - {skill: wait, args: {signal: inspection_done}}
  - {skill: transfer, args: {object: tray, from: $inspection, to: $destination}}
```

`rsf.model.workflow.expand` turns it into concrete task instances and the event
that ends the run (`request_refill`, `request_removal`). Skills (the motion) are
P3; a skill's state change is its registered **effect** (`transfer` moves the
object; skills without one change nothing).

## Extension points (`rsf.model.registry`)

| Kind | Signature | Examples |
|---|---|---|
| `generator` | `fn(name, params, ctx) -> Entity` | `rsf.box`, `rsf.frame`, `tm_tray.rack`, `tm_tray.table`, `tm_tray.pedestal` |
| `scheduler` | `cls(world, sets, **params)` with `.next(occupancy)` | `rsf.ordered` |
| `effect` | `fn(occupancy, args)` | `transfer` |
| `check` | `fn(world) -> list[Check]` | `rsf.unique_solid_names`, `rsf.robot_model`, `tm_tray.*` |
| `skill` | P3 | |

A new cell type (another rack design, a conveyor) is a plugin with its own
generators; nothing in `rsf` changes.

## The TM tray port and how it is checked

`cells/tm_tray/plugin/tm_tray` ports AI-Robotics-V2 `cell_core` (hardware
constants, frames, slots, scene generators, audit). A frozen copy of the
original is in `tests/reference/ai_robotics_v2`; the tests call it directly:

- all rack, table, pedestal and floor boxes, all 481 jig boxes, all 37 slot frames,
  the robot base pose and the AprilGrid ids equal the original to 1e-12 for
  `as_built`, `recommended` and `strict`;
- the tray (now rails + recessed body) fills exactly the original tray box;
- the workflow expansion follows the original `SequentialScheduler`;
- the audit's statuses and values equal the original's for every ported check;
- the cuRobo scene equals `cell_core.scene.to_curobo_scene`.

**Deliberate difference**: racks are built at their measured width. The
original Isaac builder applied one width margin to both racks, making the
incoming shelf 0.816 m instead of 0.800 m (D22); the tests reproduce the
original exactly by overriding the width, and pin that this is the only
difference.

Not ported (no longer applicable): `config_valid` (an invalid layout does not
build), `generator_fidelity` and `env_adapter` (the Isaac stage will be compiled
from the World in P5), `protocol` (P5/drivers), `locating` and `sequence` (P3/P6).

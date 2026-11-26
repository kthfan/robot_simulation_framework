# Cell model: assets, layout, workflow

A cell is three kinds of file, all under `cells/<cell>/`:

| File | Says | Format |
|---|---|---|
| `assets/` (+ `asset_spec.yaml`) | what the hardware is: robot, tool, fixtures, objects | URDF + meshes |
| `layouts/*.yaml` | where everything stands in this installation | `rsf.layout/1` |
| `workflows/*.yaml` | what the cell does, in what order | `rsf.workflow/1` |
| `plugin/<name>/` | cell-specific generators, checks, skills, schedulers | Python package |

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
process: {standoff_near: 0.2768}  # free-form for the plugin; the framework reads motion, continuity, judge
                                  # (docs/new_cell.md §7)
entities:                         # built in order; later entities may use earlier features
  pedestal: {generator: tm_tray.pedestal, params: {...}}
  incoming: {generator: tm_tray.rack, params: {...}}
  fence:    {urdf: ../assets/fence.urdf, pose: {xyz: [0, 1, 0], rpy: [0, 0, 0]}}
placements:                       # a fixed copy at every selected feature
  jig: {urdf: ../assets/tray_jig_with_label/tray_jig_with_label.urdf, at: {kind: slot}}
objects:                          # movable, a template at each selected feature (mass: the URDF's)
  tray: {urdf: ../assets/tray/tray.urdf, at: {kind: slot}, offset: {xyz: [0, 0, 0.035]}, initially: all}
robot: {urdf: ../assets/tm5s_egu50.urdf, base: {feature: pedestal/arm_base}, tool_frame: flange}
audit: ["tm_tray.*", "rsf.*"]     # optional: which checks run (default all)
```

- **Poses**: `{xyz, rpy}`, `{xyz, quat_wxyz}`, `{matrix}`, or `{feature: key, offset: pose}`.
- **Overrides**: dotted paths into the resolved document
  (`entities.pedestal.params.leg_height`): `--set path=value` on the command
  line, `load_world(layout, {"entities.pedestal.params.leg_height": 0.6})` in
  Python. A path that does not exist is an error, so a typo cannot silently do
  nothing. Variants are small files that `extends:` a record (`strict.yaml`
  changes one number of `as_built.yaml`).
- **Paths** are relative to the file that wrote them, also across `extends`.
- URDF meshes that are exact axis-aligned boxes become box solids.

Built-in generators (any cell):

| Generator | `params` | Builds |
|---|---|---|
| `rsf.box` | `size: [x, y, z]` (full size, centred on the pose), `pose`, `tags: [...]`, `kind` (default `static`) | one box solid named after the entity, tagged with its name and `tags` |
| `rsf.frame` | `pose`, `kind` (default `frame`), `attrs: {...}` | one feature named after the entity, no solid |

## Selectors

Used by layouts (`at:`) and workflows (`sets:`):
`"incoming/L0/*"` (key glob), `{kind: slot, role: outgoing, layer: [0, 3], z: {min: 1.0}}`
(all must match; list = one of; `{min,max}` = range), or a list of selectors (union).
A selector's keys are `key`, `kind`, `entity`, `x`/`y`/`z` (the feature's position) and
any attribute in the feature's `attrs`. `order:` (schedulers) sorts by the same names;
a leading `-` sorts descending (`order: [-height, column]`: top shelf first, then by column).

## Workflow (`rsf.workflow/1`)

```yaml
schema: rsf.workflow/1
name: inspect_and_sort
sets: {incoming: {kind: slot, role: incoming}, outgoing: {...}, inspection: {...}}
initial: {tray: [incoming]}                  # occupancy to plan from
scheduler:
  type: rsf.ordered
  params: {bind: {source:      {set: incoming, order: [-height, column], state: occupied, on_none: request_refill},
                  destination: {set: outgoing, order: [height, column],  state: empty,    on_none: request_removal}}}
cycle:
  - {skill: tm_tray.transfer, effect: transfer, args: {object: tray, from: $source, to: $inspection}}
  - {skill: wait, args: {signal: inspection_done}}
  - {skill: tm_tray.transfer, effect: transfer, args: {object: tray, from: $inspection, to: $destination}}
```

`rsf.model.workflow.expand` (`python -m rsf expand`) turns it into concrete
task instances, cycle by cycle, and the event that ends the run.

| Top-level key | Meaning |
|---|---|
| `schema`, `name`, `description` | identification (`schema` must be `rsf.workflow/1` if given) |
| `plugins` | plugin paths, as in a layout (usually the layout already loads them) |
| `sets` | `{name: selector}`: named feature sets; each must match at least one feature |
| `initial` | `{object: [set names or feature keys]}`: where objects start (default: the layout's `initially`) |
| `scheduler` | `{type, params}`: which features each cycle works on (below) |
| `cycle` | the steps repeated every cycle |
| `params` | passed to every skill |
| `max_cycles` | stop after this many cycles (default 1000); the end event is then `max_cycles` |

A **step** is `{skill, args, effect}`:

- `skill`: a registered skill (`<plugin>.<name>`). A name that is not registered
  (`wait`) is a step without motion: the trajectory gets a `hold:<skill>`.
- `args`: passed to the skill. `$name` is a scheduler binding, or a set that
  holds exactly one feature (`$inspection`).
- `effect`: the bookkeeping the step does: `transfer` moves the object at
  `from` to `to`. Without `effect:`, the effect registered under the skill's
  own name is used, if any. Plugin skills are named `<plugin>.<name>`, so
  **a step that moves something must say `effect: transfer`**. `effect: none`
  means none.

`expand` raises instead of looping when a cycle would start in the same
occupancy with the same bindings as the cycle before. That happens with a step
missing its `effect:`, or with a round trip under `rsf.ordered`. A workflow
that repeats on purpose sets `max_cycles`.

### Schedulers

A scheduler is `register("scheduler", name)`, constructed as
`cls(world, sets, **params)` (`sets`: `{name: [features]}`). Its
`.next(occupancy)` returns the cycle's bindings `{name: feature key}`, or an
`Event(name, binding, detail)` (`rsf.orchestrator.scheduler.Event`) that ends
the run. Built in:

| Type | Binds | Ends |
|---|---|---|
| `rsf.ordered` | each binding (`set`, `order`, `state`: `occupied` / `empty` / `any`, `on_none`) to the first feature in `order` whose state matches. No memory between cycles: the run ends when the effects drain a set | `on_none` of the binding that found nothing (default `<binding>_exhausted`) |
| `rsf.each` | exactly one binding (`set`, `order`, `state`) to the next feature not yet visited (a feature in the wrong state is skipped). For round trips and inspections, where the occupancy comes back | `on_done` (default `all_visited`) |

Another pattern (priorities, a feed from a sensor) is a scheduler in the cell's
plugin; see `rsf.orchestrator.scheduler` for both built-ins.

## Extension points (`rsf.model.registry`)

| Kind | Signature | Examples |
|---|---|---|
| `generator` | `fn(name, params, ctx) -> Entity` | `rsf.box`, `rsf.frame`, `tm_tray.rack`, `tm_tray.table`, `tm_tray.pedestal`, `example.stand` |
| `check` | `fn(world) -> list[Check]` | `rsf.unique_solid_names`, `rsf.robot_model`, `tm_tray.*`, `example.stands` |
| `skill` | `fn(world, args) -> list[Sequence]` (P3, docs/new_cell.md §5.3) | `tm_tray.pick/place/transfer`, `example.pick/place/transfer` |
| `scheduler` | `cls(world, sets, **params)` with `.next(occupancy)` | `rsf.ordered` |
| `effect` | `fn(occupancy, args)` | `transfer` |
| `feature_move` | `fn(feature, T) -> Feature` (rebuild a moved feature) | `slot` (tm_tray) |
| `targets` | `fn(world, feature) -> {name: T}` (reach studies) | `tm_tray.grasp`, `tm_tray.cycle` |
| `vision` | cell-specific locating inputs | `tm_tray.labels`, `tm_tray.camera` |

A new cell type (another rack design, a conveyor) is a plugin with its own
generators, checks and skills; nothing in `rsf` changes. docs/new_cell.md walks
through one (`cells/example_pick_place`).

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
build), `generator_fidelity` and `env_adapter` (the Isaac stage is compiled from
the World, `rsf.compile.usd`), `protocol` (real-robot drivers, not done yet);
`locating` and `sequence` became P6 (`tools/locate_slots.py`) and the tray
plugin's `sequence.py`.

# File formats

Units are always metres, radians and seconds, except where a field name says
otherwise (`_mm`, `_deg`, the CSV's columns). Poses are 4×4 row-major
matrices, or `xyz` plus `quat_wxyz`. Everything is in the world frame unless a
field says otherwise.

| File | Schema | Written by | Read by |
|---|---|---|---|
| layout YAML | `rsf.layout/1` | you | everything (`rsf.model.layout.load_world`) |
| workflow YAML | `rsf.workflow/1` | you | `rsf.model.workflow` |
| teach table JSON + CSV | (below) | `python -m rsf teach` | `tools/plan_workflow.py` |
| cuRobo job | `rsf.curobo_job/1` | `plan_workflow.py job` | `tools/kaggle/curobo_free_plan.py` |
| cuRobo results | (below) | `curobo_free_plan.py` | `plan_workflow.py assemble --paths` |
| trajectory | `rsf.trajectory/1` | `plan_workflow.py assemble` | `tools/sim_workflow.py`, the real-robot exporter (to be designed) |
| trajectory summary | `rsf.trajectory_summary/1` | `plan_workflow.py assemble` | people, tests (committed; the trajectory is not) |
| judge report JSON | (below) | `tools/sim_workflow.py --out` | people, tests |
| execution report `.npz` | (below) | `tools/sim_workflow.py --report` | analysis scripts |
| live view API | `rsf.live/1` | `--live` | browsers, programs (docs/live_view.md) |

## Layout and workflow

See [cell_model.md](cell_model.md) (keys, selectors, poses, overrides) and the
commented example in `cells/example_pick_place/`. Both loaders reject unknown
top-level keys.

## Teach table (`python -m rsf teach --out <prefix>`)

`<prefix>.json`:

```json
{"ok": true,
 "meta": {"layout": "...", "overrides": {}, "robot": "<urdf>", "process": {...},
          "workflow": "...", "tasks": [{"index": 0, "cycle": 0, "skill": "...", "args": {...},
                                        "sequences": ["pick:stand_a/top", "place:stand_c/top"]}],
          "end": {"name": "sources_empty", "detail": "..."},
          "prefer": {"branch": {"joint_3": -1, "joint_5": 1}, "q_ref": [...]}, "seconds": 4.0},
 "sequences": [
   {"sequence": "pick:stand_a/top", "ok": true, "problems": [], "meta": {"skill": "...", "slot": "..."},
    "points": [
      {"sequence": "pick:stand_a/top", "name": "grasp",
       "q": [6 joint values, rad], "xyz": [m], "quat_wxyz": [...],     // tool frame pose, world
       "joints": {"finger_right_joint": 0.035, ...},                   // other joints commanded here
       "arrive": "line",
       "clearance": 0.010, "margin": 0.005, "nearest": ["finger_left", "stand_a"],   // smallest-slack pair here
       "along_slack": 0.005, "along_samples": 20,                      // worst slack along the arrival (line/path)
       "manipulability": 0.080, "limit_margin": 0.91,                  // Yoshikawa; rad to the nearest operating limit
       "branch": {"joint_3": -1, "joint_5": 1}, "action": "close the fingers"}]}]}
```

`meta.tasks` and `meta.end` are present only with `--workflow`. A sequence
that failed has `ok: false`, an empty `points` and the reasons in `problems`. A
taught variant in another arm configuration is named
`<sequence>#joint_3+1,joint_5-1` and has `meta.variant_of`.

`<prefix>.csv` has one row per point, in degrees and millimetres: `sequence,
point, arrive, j1_deg … j6_deg, x_mm, y_mm, z_mm, qw, qx, qy, qz, jaw,
clearance_mm, nearest, along_slack_mm, manipulability, limit_margin_deg,
action`.

## cuRobo job (`rsf.curobo_job/1`)

```json
{"schema": "rsf.curobo_job/1", "robot_urdf": "<path relative to the repo root>",
 "base": [[4x4]], "tool_frame": "flange", "limit_margin_deg": 5.0, "mesh_collision_margin": 0.005,
 "scenes": {"<id>": {cuRobo world dict, robot base frame}},
 "segments": [{"name": "free003:...", "q_start": [...], "q_goal": [...], "lock_joints": {"jaw": 0.03},
               "scene": "<id>", "carried": {"link": "...", "boxes": [{"name", "dims", "pose_in_link"}]} | null,
               "tool_bound": {"tool_axis": [...], "reference_axis": [0, 0, 1], "max_angle": rad} | null}],
 "meta": {...}}
```

Results: `{"<segment name>": {"success": true, "q": [[joints] ...], "dt": s}, ...}`.
A result whose ends differ from the segment's by more than 1e-3 rad is
discarded as stale.

## Trajectory (`rsf.trajectory/1`)

```json
{"schema": "rsf.trajectory/1",
 "joint_names": ["joint_1", ..., "joint_6", "<other joints, sorted>"],
 "t": [s], "q": [[rad / m per joint] per sample], "qd": [[per joint] per sample],
 "segments": [{"name": "pick:stand_a/top/grasp", "kind": "line", "t0": 3.83, "t1": 6.03,
               "caps": "near", "event": ""}],
 "meta": {"layout", "workflow", "cycles", "teach", "paths", "free_source", "problems": [], "ok": true, "seconds"}}
```

Samples are every `process.motion.dt`. Segments run rest to rest. Segment
names:

| name | meaning |
|---|---|
| `free<NNN>:<sequence>/<waypoint>` | free move to a waypoint |
| `free<NNN>:home` | free move back to `motion.home` |
| `<sequence>/<waypoint>` | line or path arrival |
| `<sequence>/<waypoint>:act` | hold while the other joints move (the waypoint's `action`, in `event`) |
| `hold:<skill>` | a workflow step without motion (e.g. `wait`) |

`<plan>.summary.json` (`rsf.trajectory_summary/1`) has `duration`,
`samples`, `segments` and `meta`, without the samples.

## Judge report (`tools/sim_workflow.py --out`)

```json
{"ok": true,
 "transfers": [{"src": "stand_a/top", "dst": "stand_c/top", "object": "block/stand_a/top",
                "t0": 3.8, "t1": 17.6, "carry": [t_start, t_end],
                "seat_mm": 0.02, "seat_deg": 0.0, "slip_mm": 0.98, "track_mm": 0.15, "carry_tilt_deg": 0.02,
                "contacts": [{"link", "other", "n", "t_first", "t_last", "depth_mm"}],
                "caps_ok": true, "carry_kin": {"v", "a", "w", "alpha", "a_down"},
                "checks": {"placement": true, "carry_tilt": true, "slip": true, "contacts": true,
                           "tracking": true, "caps": true},
                "ok": true}],
 "other_contacts": [...],                        // robot contacts outside every transfer
 "meta": {"layout", "workflow", "plan", "sim_seconds", "wall_seconds", "backend", "params": {...},
          "live_debug": {...}}}                   // live_debug only when debug code ran (the run is not an acceptance result)
```

## Execution report (`--report run.npz`)

NumPy arrays sampled every `SimParams.sample_dt`: `t`, `flange` and
`flange_planned` (n×4×4), `q` (n × trajectory joints), `object_names` and
`objects` (k×n×4×4). `json` holds a JSON string with `contacts` (`{t, link,
other, depth}`), `segments` and `meta`. Load it with `tools/sim_workflow.py`'s
`save_report` counterpart: `np.load(path)` and `json.loads(d["json"][()])`.

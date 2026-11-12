"""Working-cell core: geometry, configuration, and task sequencing.

The single editable source for everything both repos need to agree on. Pure
Python — no ROS, no GPU, no Isaac Sim — so that it imports on any host,
including the air-gapped ones. ``tools/sync_shared.py`` vendors it into each
consumer repo and each repo fails its tests if that copy drifts (D8).

Layering, innermost first::

    assets      measured hardware dimensions; never varies by deployment
    config      cell layout; varies by deployment, lives in YAML
    frames      pose construction and URDF/USD convention conversion
    geometry    slots, clearances, tolerance budget, derived from both
    sequence    the flange poses that service a slot
    scheduler   the order slots are serviced in
    env_adapter translates a config into Isaac Sim builder arguments
    audit       checks any config against all of the above, and reports

Typical use::

    from cell_core import load_preset, enumerate_slots, grasp_sequence

    cfg = load_preset("recommended")
    for slot in enumerate_slots(cfg):
        poses = grasp_sequence(cfg, slot)
"""
from .assets import FOCAL_CONFIGS, focal_config_for

# Only the names that cannot collide with the submodule itself. ``audit()`` is
# deliberately not re-exported here: binding it at package level would replace
# the ``cell_core.audit`` module object, and ``import cell_core.audit`` would
# then hand back a function. Call it as ``cell_core.audit.audit(cfg)``.
from .audit import Check, failures, skipped, verdict, warnings
from .config import (
    CellConfig,
    CellConfigError,
    RackConfig,
    RobotMount,
    TableConfig,
    SCAN_PARAMS,
    apply_override,
    available_presets,
    load_config,
    load_preset,
    replace_mount,
    save_config,
    validate,
)
from .env_adapter import generator_kwargs, load_generator, unsupported
from .frames import flange_correction, flange_rotation, make_pose, usd_to_urdf
from .geometry import (
    Slot,
    enumerate_slots,
    lower_layer_clearance,
    plate_free_standoff,
    shoulder_z,
    tolerance_budget,
)
from .scheduler import Scheduler, SequentialScheduler
from .sequence import GRASP_PHASES, grasp_sequence

__all__ = [
    "CellConfig", "CellConfigError", "RackConfig", "RobotMount", "TableConfig",
    "SCAN_PARAMS", "apply_override",
    "available_presets", "load_config", "load_preset", "replace_mount",
    "save_config", "validate",
    "Check", "failures", "skipped", "verdict", "warnings",
    "FOCAL_CONFIGS", "focal_config_for",
    "generator_kwargs", "load_generator", "unsupported",
    "flange_correction", "flange_rotation", "make_pose", "usd_to_urdf",
    "Slot", "enumerate_slots", "lower_layer_clearance", "plate_free_standoff",
    "shoulder_z", "tolerance_budget",
    "Scheduler", "SequentialScheduler",
    "GRASP_PHASES", "grasp_sequence",
]

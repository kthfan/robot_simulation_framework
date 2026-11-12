"""Geometry checks that can be run on any configuration, on any machine.

Every check here is pure arithmetic over :mod:`cell_core.config`,
:mod:`cell_core.assets` and :mod:`cell_core.geometry`. Nothing in this module
needs a robot, a URDF, a solver or a simulator. That is deliberate: it means the
whole audit runs in milliseconds on a laptop, and the expensive reachability
sweep can be added on top by whoever has EAIK installed.

Two rules govern the design:

*   **An audit never raises.** A crash cannot be compared against a baseline. A
    broken configuration must come back as a report saying *which* check failed,
    so that the difference between last week and this week is always a diff of
    two reports rather than a traceback.
*   **A skipped check is not a passing check.** Anything that could not be
    evaluated is recorded as ``skip`` and listed by :func:`skipped`, so a green
    verdict can always be read alongside what it did not look at.

Statuses:

``pass``
    The check was evaluated and the cell satisfies it.
``fail``
    The check was evaluated and the cell does not satisfy it. The cell cannot be
    built or operated as configured.
``warn``
    The cell works, but something is worth knowing: a margin is thin, a number
    is unconfirmed, or a downstream tool cannot reproduce the configuration
    faithfully.
``skip``
    The check could not be evaluated.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from . import assets as A
from . import config as C
from . import geometry as G

PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"

#: Every status an audit may report, worst first.
STATUSES = (FAIL, WARN, SKIP, PASS)

#: Minimum grid width in pixels for a pose estimate worth trusting.
#:
#: This is a floor, not a specification. Nobody has measured the pose error of
#: the AprilGrid detector as a function of grid pixels on this hardware; until
#: somebody does, this number exists only to catch a standoff that is absurdly
#: far. Both shipped presets clear it by a factor of five. Treat a failure here
#: as "you have moved the camera much further than anyone intended", not as
#: "the pose estimate is now out of tolerance".
MIN_GRID_PIXELS = 150.0

#: Smallest pitch tolerance, in degrees, worth calling a tolerance.
#:
#: Below this the plate would have to be aligned better than the arm's own
#: repeatability, which means the insertion is not controlled by the corridor
#: any more but by luck.
MIN_PITCH_DEG = 0.5


@dataclass(frozen=True)
class Check:
    """One question asked of a configuration, and the answer.

    ``value`` carries the underlying number when there is one, so a scan can
    plot how a margin moves without re-deriving it from the text.
    """

    name: str
    status: str
    detail: str
    value: float = None

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _check(name, ok, detail_ok, detail_bad, value=None, soft=False) -> Check:
    """Build a check from a boolean. *soft* downgrades a failure to a warning."""
    if ok:
        return Check(name, PASS, detail_ok, value)
    return Check(name, WARN if soft else FAIL, detail_bad, value)


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────
def _check_config(cfg) -> Check:
    """Does the configuration satisfy its own schema?

    Everything downstream assumes it does, so this runs first and the rest of
    the audit is guarded on it.
    """
    try:
        C.validate(cfg)
    except C.CellConfigError as exc:
        return Check("config_valid", FAIL, f"configuration is not valid: {exc}")
    return Check("config_valid", PASS, "configuration satisfies the schema")


def _check_rack_width(rack) -> Check:
    need = C.min_rack_width(rack.n_jigs, rack.jig_gap)
    slack = rack.width - need
    return _check(
        f"rack_width/{rack.name}",
        slack >= 0,
        f"{rack.n_jigs} jigs need {need:.3f} m, shelf is {rack.width:.3f} m "
        f"({slack * 1000:.0f} mm spare)",
        f"{rack.n_jigs} jigs need {need:.3f} m but the shelf is only "
        f"{rack.width:.3f} m ({-slack * 1000:.0f} mm short)",
        value=slack,
    )


def _check_layer_gap(rack) -> Check:
    gaps = [b - a for a, b in zip(rack.layer_heights, rack.layer_heights[1:])]
    if not gaps:
        return Check(f"layer_gap/{rack.name}",
                     PASS, "single layer, no gap to check")
    tightest = min(gaps)
    slack = tightest - C.MIN_LAYER_GAP
    return _check(
        f"layer_gap/{rack.name}",
        slack >= 0,
        f"tightest layer gap {tightest:.3f} m, needs {C.MIN_LAYER_GAP:.3f} m "
        f"({slack * 1000:.0f} mm spare)",
        f"tightest layer gap {tightest:.3f} m but a tray plus its corridor "
        f"needs {C.MIN_LAYER_GAP:.3f} m ({-slack * 1000:.0f} mm short)",
        value=slack,
    )


def _check_standoff_plate_free(cfg) -> Check:
    """At the close read, is the plate still outside the shelf?

    The plate sticks out 212 mm ahead of the camera. If the camera reads closer
    than that, the plate is already inside the shelf opening while the arm is
    still servoing on the label, and anything that moves the arm sideways scrapes
    the tray below.
    """
    need = G.plate_free_standoff()
    slack = cfg.standoff_near - need
    return _check(
        "standoff_plate_free",
        slack >= 0,
        f"close read at {cfg.standoff_near:.4f} m keeps the plate outside the "
        f"shelf (needs {need:.4f} m, {slack * 1000:.1f} mm spare)",
        f"close read at {cfg.standoff_near:.4f} m puts the plate "
        f"{-slack * 1000:.1f} mm inside the shelf; it must be at least "
        f"{need:.4f} m",
        value=slack,
    )


def _check_grid_in_view(cfg, name, standoff) -> Check:
    """Does the whole AprilGrid fall inside the frame?

    The grid is wider than it is tall, so width is the binding dimension.
    """
    half = G.fov_half_width(standoff)
    slack = half - A.GRID_W / 2.0
    return _check(
        f"grid_in_view/{name}",
        slack >= 0,
        f"at {standoff:.4f} m the frame is {2 * half:.3f} m wide and the grid "
        f"is {A.GRID_W:.3f} m ({slack * 1000:.0f} mm spare each side)",
        f"at {standoff:.4f} m the frame is only {2 * half:.3f} m wide but the "
        f"grid is {A.GRID_W:.3f} m; it is cropped by {-slack * 1000:.0f} mm "
        f"each side",
        value=slack,
    )


def _check_grid_pixels(cfg, name, standoff) -> Check:
    px = G.grid_pixels(standoff)
    return _check(
        f"grid_pixels/{name}",
        px >= MIN_GRID_PIXELS,
        f"grid spans {px:.0f} px at {standoff:.4f} m "
        f"(floor {MIN_GRID_PIXELS:.0f} px)",
        f"grid spans only {px:.0f} px at {standoff:.4f} m, below the "
        f"{MIN_GRID_PIXELS:.0f} px floor; the camera is too far to read it",
        value=px,
    )


def _check_tolerance_pitch(cfg) -> Check:
    """How far can the plate be off-axis and still enter the corridor?

    Computed on the flat corridor rather than the recessed one: the recess is a
    bonus that only exists where the tray actually has it, and an insertion
    budget should not depend on a feature the audit cannot confirm is present.
    """
    budget = G.tolerance_budget(use_recess=False)
    pitch = budget["pitch_deg"]
    return _check(
        "tolerance_pitch",
        pitch >= MIN_PITCH_DEG,
        f"plate may pitch {pitch:.2f}° and still enter the corridor",
        f"plate may pitch only {pitch:.2f}°, below the {MIN_PITCH_DEG:.2f}° "
        f"floor; the corridor is too tight for the plate",
        value=pitch,
    )


def _check_table_confirmed(cfg) -> Check:
    return _check(
        "table_confirmed",
        cfg.table.confirmed,
        f"table position measured: {cfg.table.face_distance:.3f} m out, "
        f"{cfg.table.height:.3f} m up",
        f"table position is a placeholder ({cfg.table.face_distance:.3f} m out, "
        f"{cfg.table.height:.3f} m up) and has not been measured; every table "
        f"result below is provisional",
        soft=True,
    )


def _check_mount_height(cfg) -> Check:
    """Is the arm high enough to service the whole cell?

    The threshold comes from the reachability sweep, not from a drawing: below
    it the two top-back outgoing slots fall outside the arm's reach. It is
    recorded as a warning rather than a failure because the sweep itself is the
    authority, and it may be re-run with a different rack distance that moves
    this number.
    """
    z = cfg.robot.origin_z
    slack = z - C.ORIGIN_Z_THRESHOLD
    return _check(
        "mount_height",
        slack >= 0,
        f"shoulder mounted at {z:.4f} m, above the {C.ORIGIN_Z_THRESHOLD:.3f} m "
        f"full-service threshold ({slack * 1000:.0f} mm spare)",
        f"shoulder mounted at {z:.4f} m, {-slack * 1000:.0f} mm below the "
        f"{C.ORIGIN_Z_THRESHOLD:.3f} m threshold; expect the top-back outgoing "
        f"slots to be unreachable",
        value=slack,
        soft=True,
    )


def _check_generator_fidelity(cfg) -> Check:
    """Can the scene generator actually build this cell?

    The generator takes one shelf width for both racks. A configuration with two
    different widths is still a valid cell, but the simulation of it will not be
    the cell that was configured, and a result measured in that simulation does
    not transfer.
    """
    try:
        from . import env_adapter
        problems = env_adapter.unsupported(cfg)
    except Exception as exc:                       # pragma: no cover - defensive
        return Check("generator_fidelity", SKIP,
                     f"could not evaluate generator fidelity: {exc}")
    if not problems:
        return Check("generator_fidelity", PASS,
                     "the scene generator can build this cell exactly")
    return Check("generator_fidelity", WARN,
                 "the simulated cell will differ from this configuration: "
                 + "; ".join(problems))


# ─────────────────────────────────────────────────────────────────────────────
# The audit
# ─────────────────────────────────────────────────────────────────────────────
def audit(cfg) -> list:
    """Run every geometry check against *cfg* and return the results.

    Never raises. A configuration so broken that the later checks cannot even be
    attempted comes back as a failed ``config_valid`` plus a ``skip`` for each
    check that could not run, which is still a report you can diff.
    """
    checks = [_check_config(cfg)]

    for name, fn in (
        ("rack_width", _check_rack_width),
        ("layer_gap", _check_layer_gap),
    ):
        for rack in cfg.racks:
            checks.append(_guard(f"{name}/{rack.name}", fn, rack))

    checks.append(_guard("standoff_plate_free", _check_standoff_plate_free, cfg))
    for label, standoff in (("near", cfg.standoff_near), ("far", cfg.standoff_far)):
        checks.append(_guard(f"grid_in_view/{label}",
                             _check_grid_in_view, cfg, label, standoff))
        checks.append(_guard(f"grid_pixels/{label}",
                             _check_grid_pixels, cfg, label, standoff))

    checks.append(_guard("tolerance_pitch", _check_tolerance_pitch, cfg))
    checks.append(_guard("table_confirmed", _check_table_confirmed, cfg))
    checks.append(_guard("mount_height", _check_mount_height, cfg))
    checks.append(_guard("generator_fidelity", _check_generator_fidelity, cfg))
    return checks


def _guard(name, fn, *args) -> Check:
    """Run one check, turning any crash into a reportable skip."""
    try:
        return fn(*args)
    except Exception as exc:
        return Check(name, SKIP, f"could not be evaluated: {exc}")


def verdict(checks) -> str:
    """``"fail"`` if anything failed, otherwise ``"pass"``.

    Warnings and skips do not fail the audit — but a caller that reports a pass
    without also reporting :func:`skipped` is hiding something.
    """
    return FAIL if any(c.status == FAIL for c in checks) else PASS


def skipped(checks) -> list:
    """Names of the checks that could not be evaluated."""
    return [c.name for c in checks if c.status == SKIP]


def failures(checks) -> list:
    """The checks that failed, in the order they were run."""
    return [c for c in checks if c.status == FAIL]


def warnings(checks) -> list:
    """The checks that passed but deserve a second look."""
    return [c for c in checks if c.status == WARN]


def to_dict(checks) -> list:
    """Serialisable form, for reports that get diffed against a baseline."""
    return [dataclasses.asdict(c) for c in checks]

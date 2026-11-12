"""Drive the Isaac Sim environment builder from a :class:`CellConfig`.

The point of this module is that raising the robot (D19) must be a preset swap,
not a code edit. ``create_environment`` is configured by a dict produced by
``env_config_generator.generate_env_config``; this translates a ``CellConfig``
into that dict's keyword arguments, so the sim and the kinematic model are
built from one description.

Nothing here imports Isaac Sim. The generator is plain arithmetic, so the
translation can be unit-tested on any machine, which is the only reason it is
worth having as a separate layer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .config import CellConfig, CellConfigError, min_rack_width

#: Filename of the generator in the isaacsim repo.
GENERATOR_FILENAME = "env_config_generator.py"

#: Directories searched for the generator, relative to each ancestor of this
#: package. Ordered most-specific first.
GENERATOR_SEARCH = (
    "build_env",
    "tools",
    "todo/resources/isaacsim/build_env",
)


def load_generator(path=None):
    """Import ``generate_env_config`` from wherever the generator lives.

    Loaded by file path rather than by package name: this module is vendored
    into two repos that lay their source out differently, and neither puts the
    generator on the import path.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        here = Path(__file__).resolve()
        candidates = [anc / sub / GENERATOR_FILENAME
                      for anc in here.parents
                      for sub in GENERATOR_SEARCH]

    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_cell_core_env_generator", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_env_config

    raise FileNotFoundError(
        f"could not find {GENERATOR_FILENAME}; pass its path explicitly")


def _shared(cfg: CellConfig, field: str):
    """Read a rack property that the generator only lets us set once."""
    values = {getattr(rack, field) for rack in cfg.racks}
    if len(values) != 1:
        raise CellConfigError(
            f"the environment generator takes a single {field} for both racks, "
            f"but this config has {sorted(values)}. Either make them equal or "
            f"extend the generator.")
    return values.pop()


def _size_x_margin(cfg: CellConfig) -> float:
    """The one width allowance the generator accepts, sized for the worst rack.

    The generator derives shelf width as ``min_rack_width(n_jigs) + margin`` and
    applies the same margin to both racks, but the two real racks are not the
    same width. Taking the maximum means no shelf is ever built narrower than
    the one that was measured; :func:`unsupported` reports the racks that come
    out wider.
    """
    gap = _shared(cfg, "jig_gap")
    radius = _shared(cfg, "cylinder_radius")
    return max(rack.width - min_rack_width(rack.n_jigs, gap, radius)
               for rack in cfg.racks)


def _rack_height(rack) -> float:
    """Shelf post height: tall enough to carry its highest plate."""
    return max(rack.layer_heights) + rack.plane_thickness


def _rack_origin_distance(rack) -> float:
    """Where the generator must put a rack for its FRONT face to be at ``face_distance``.

    The generator calls this argument the front face, but the builder uses it as
    the rack's local origin, which is a corner. After the 90 degree turn that
    corner is at the BACK of the rack, so the face lands one depth nearer the
    robot. Measured on the built stage, 2026-09-25: face at 0.484, not 0.792.
    """
    return rack.face_distance + rack.depth


def generator_kwargs(cfg: CellConfig, **overrides) -> dict:
    """Translate *cfg* into ``generate_env_config`` keyword arguments.

    The incoming rack is the +X ("right") shelf and the outgoing rack the -X
    ("left") one, matching the approach vectors in the presets.
    """
    margin = _size_x_margin(cfg)
    kwargs = dict(
        dist_right_rack=_rack_origin_distance(cfg.incoming),
        dist_left_rack=_rack_origin_distance(cfg.outgoing),
        dist_table=cfg.table.face_distance,

        jig_gap=_shared(cfg, "jig_gap"),
        cylinder_radius=_shared(cfg, "cylinder_radius"),
        n_jigs_right=cfg.incoming.n_jigs,
        n_jigs_left=cfg.outgoing.n_jigs,
        right_rack_planes=list(cfg.incoming.layer_heights),
        left_rack_planes=list(cfg.outgoing.layer_heights),

        rack_depth=_shared(cfg, "depth"),
        right_rack_height=_rack_height(cfg.incoming),
        left_rack_height=_rack_height(cfg.outgoing),
        size_x_margin=margin,

        table_height=cfg.table.height,
        table_jig_x=cfg.table.jig_x,
        table_size_y=cfg.table.depth,

        robot_leg_height=cfg.robot.leg_height,
    )
    kwargs.update(overrides)
    return kwargs


def unsupported(cfg: CellConfig) -> list:
    """Everything about *cfg* the generator cannot reproduce exactly.

    Returned rather than raised: none of these stop a sim from being built,
    but all of them mean the sim is not quite the cell that was measured, and
    that should be visible rather than discovered later.
    """
    notes = []
    kwargs = generator_kwargs(cfg)
    gap, radius = kwargs["jig_gap"], kwargs["cylinder_radius"]
    margin = kwargs["size_x_margin"]

    for rack in cfg.racks:
        built = min_rack_width(rack.n_jigs, gap, radius) + margin
        if abs(built - rack.width) > 1e-9:
            notes.append(
                f"{rack.name}: shelf width will be {built:.4f} m, not the "
                f"configured {rack.width:.4f} m ({1000 * (built - rack.width):+.1f} mm). "
                f"The generator applies one width margin to both racks.")

    if not cfg.table.confirmed:
        notes.append(
            f"table: face distance {cfg.table.face_distance:.3f} m and height "
            f"{cfg.table.height:.3f} m are unverified generator defaults, not "
            f"measurements of the real cell.")

    return notes

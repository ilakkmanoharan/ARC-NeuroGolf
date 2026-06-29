"""Phase 21 — dynamic bbox-relative gravity ONNX solvers."""

from __future__ import annotations

import numpy as np

from arc_genome.genome.ops.arcgen_gather import _arcgen_examples
from arc_genome.genome.compose.primitives import (
    _gravity_down,
    _gravity_left,
    _gravity_right,
    _gravity_up,
)
from arc_genome.onnx.gravity import build_gravity_model, max_grid_extent


def _pairs(td) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array(ex["input"]), np.array(ex["output"])) for ex in _arcgen_examples(td)]


def _s_gravity_dynamic(td, direction: str, rule_fn):
    pairs = _pairs(td)
    if not pairs:
        return None
    if not all(np.array_equal(rule_fn(inp), out) for inp, out in pairs):
        return None
    mh, mw = max_grid_extent(td)
    return build_gravity_model(direction, mh, mw)


def s_gravity_up_dynamic(td):
    return _s_gravity_dynamic(td, "up", _gravity_up)


def s_gravity_down_dynamic(td):
    return _s_gravity_dynamic(td, "down", _gravity_down)


def s_gravity_left_dynamic(td):
    return _s_gravity_dynamic(td, "left", _gravity_left)


def s_gravity_right_dynamic(td):
    return _s_gravity_dynamic(td, "right", _gravity_right)


ARCgen_GRAVITY_SOLVERS = [
    ("gravity_up_dynamic", s_gravity_up_dynamic),
    ("gravity_down_dynamic", s_gravity_down_dynamic),
    ("gravity_left_dynamic", s_gravity_left_dynamic),
    ("gravity_right_dynamic", s_gravity_right_dynamic),
]

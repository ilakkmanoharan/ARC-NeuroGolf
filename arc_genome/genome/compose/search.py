"""Phase 4 composition search over grid primitives."""

from __future__ import annotations

import itertools

import numpy as np

from arc_genome.data.encoding import get_examples
from arc_genome.genome.compose.primitives import PRIMITIVE_NAMES, apply_chain
from arc_genome.genome.ops.analytical import s_spatial_gather


def _chains(depth: int):
    for d in range(1, depth + 1):
        for chain in itertools.product(PRIMITIVE_NAMES, repeat=d):
            if d > 1 and chain[0] == "identity":
                continue
            yield chain


def solve_composition(td, max_depth: int = 3):
    """Find a primitive chain matching all I/O pairs; emit via spatial_gather."""
    exs = get_examples(td)
    for chain in _chains(max_depth):
        try:
            if all(np.array_equal(apply_chain(inp, chain), out) for inp, out in exs):
                return s_spatial_gather(td)
        except Exception:
            continue
    return None

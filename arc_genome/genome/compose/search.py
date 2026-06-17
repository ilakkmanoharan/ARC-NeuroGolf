"""Phase 4 composition search over grid primitives."""

from __future__ import annotations

import itertools

import numpy as np

from arc_genome.data.encoding import get_examples
from arc_genome.genome.compose.primitives import PRIMITIVE_NAMES, apply_chain
from arc_genome.onnx.gather import build_gather_model


def _chains(depth: int):
    for d in range(1, depth + 1):
        for chain in itertools.product(PRIMITIVE_NAMES, repeat=d):
            if d > 1 and chain[0] == "identity":
                continue
            yield chain


def _idx_from_chain(exs, chain):
    """Position-based index map for a matching primitive chain."""
    inp, out = exs[0]
    ih, iw = inp.shape
    oh, ow = out.shape
    idx = np.zeros((oh, ow, 2), dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            found = False
            for sr in range(ih):
                for sc in range(iw):
                    probe = np.zeros((ih, iw), dtype=np.int64)
                    probe[sr, sc] = 1
                    if apply_chain(probe, chain)[r, c] == 1:
                        idx[r, c] = [sr, sc]
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
    for inp, out in exs:
        pred = np.zeros((oh, ow), dtype=np.int64)
        for r in range(oh):
            for c in range(ow):
                sr, sc = idx[r, c]
                pred[r, c] = inp[sr, sc]
        if not np.array_equal(pred, out):
            return None
    return idx


def solve_composition(td, max_depth: int = 3):
    """Find a primitive chain matching all I/O pairs; emit via spatial gather."""
    exs = get_examples(td)
    for chain in _chains(max_depth):
        try:
            if not all(np.array_equal(apply_chain(inp, chain), out) for inp, out in exs):
                continue
            idx = _idx_from_chain(exs, chain)
            if idx is None:
                continue
            oh, ow = exs[0][1].shape
            return build_gather_model(oh, ow, idx)
        except Exception:
            continue
    return None

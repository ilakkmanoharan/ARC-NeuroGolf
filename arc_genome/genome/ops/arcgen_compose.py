"""Phase 20 — ARC-GEN depth-2 composition over extended grid primitives."""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

from arc_genome.config import get_config
from arc_genome.genome.compose.primitives import (
    PRIMITIVE_FUNCS,
    apply_chain_fns,
    apply_color_map,
    fit_color_map,
)
from arc_genome.genome.ops.arcgen_gather import (
    _arcgen_examples,
    s_position_gather_arcgen,
    s_spatial_gather_arcgen,
)
from arc_genome.onnx.gather import build_gather_model

GridFn = Callable[[np.ndarray], np.ndarray]


def _pairs(td) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array(ex["input"]), np.array(ex["output"])) for ex in _arcgen_examples(td)]


def _build_compose_ops(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, GridFn]:
    ops: dict[str, GridFn] = dict(PRIMITIVE_FUNCS)
    cm = fit_color_map(pairs)
    if cm is not None:
        ops["color_map"] = lambda g, m=cm: apply_color_map(g, m)

    for color in range(1, 10):

        def _keep(g, c=color):
            return np.where(g == c, c, 0)

        def _remove(g, c=color):
            return np.where(g == c, 0, g)

        ops[f"keep_{color}"] = _keep
        ops[f"remove_{color}"] = _remove

    return ops


def _chain_matches(pairs: list[tuple[np.ndarray, np.ndarray]], fns: tuple[GridFn, ...]) -> bool:
    for inp, out in pairs:
        try:
            pred = apply_chain_fns(inp, fns)
        except Exception:
            return False
        if not np.array_equal(pred, out):
            return False
    return True


def _idx_from_chain_fns(exs: list[tuple[np.ndarray, np.ndarray]], fns: tuple[GridFn, ...]):
    """Position gather index map for a matching function chain."""
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
                    try:
                        val = apply_chain_fns(probe, fns)[r, c]
                    except Exception:
                        continue
                    if val == 1:
                        idx[r, c] = [sr, sc]
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
    for inp, out in exs:
        pred = np.zeros((oh, ow), dtype=np.int64)
        for r in range(min(oh, out.shape[0])):
            for c in range(min(ow, out.shape[1])):
                sr, sc = int(idx[r, c, 0]), int(idx[r, c, 1])
                if sr < inp.shape[0] and sc < inp.shape[1]:
                    pred[r, c] = inp[sr, sc]
        if not np.array_equal(pred, out[:oh, :ow]):
            return None
    return idx, oh, ow


def _fit_gather_const_from_chain(
    pairs: list[tuple[np.ndarray, np.ndarray]], fns: tuple[GridFn, ...]
):
    """Fit gather+const by probing single-pixel inputs through the chain."""
    from arc_genome.onnx.gather import build_gather_model_with_const

    shapes = {inp.shape for inp, _ in pairs}
    if len(shapes) != 1:
        return None
    ih, iw = shapes.pop()
    oh, ow = next(out.shape for _, out in pairs)
    if (oh, ow) != (ih, iw):
        return None
    idx = np.full((oh, ow, 2), -1, dtype=np.int64)
    cst = np.full((oh, ow), -1, dtype=np.int64)

    for r in range(oh):
        for c in range(ow):
            targets = {int(out[r, c]) for _, out in pairs}
            if len(targets) != 1:
                return None
            target = targets.pop()
            if target == 0:
                cst[r, c] = 0
                continue
            src = None
            for sr in range(ih):
                for sc in range(iw):
                    probe = np.zeros((ih, iw), dtype=np.int64)
                    probe[sr, sc] = 1
                    try:
                        val = int(apply_chain_fns(probe, fns)[r, c])
                    except Exception:
                        continue
                    if val != 1:
                        continue
                    if src is not None:
                        return None
                    src = (sr, sc)
            if src is None:
                return None
            idx[r, c] = [src[0], src[1]]

    for inp, out in pairs:
        pred = np.zeros((oh, ow), dtype=np.int64)
        for r in range(oh):
            for c in range(ow):
                if idx[r, c, 0] >= 0:
                    pred[r, c] = inp[idx[r, c, 0], idx[r, c, 1]]
                elif cst[r, c] >= 0:
                    pred[r, c] = cst[r, c]
                else:
                    return None
        if not np.array_equal(pred, out):
            return None
    return build_gather_model_with_const(ih, iw, oh, ow, idx, cst)


def _chain_non_trivial(pairs: list[tuple[np.ndarray, np.ndarray]], fns: tuple[GridFn, ...]) -> bool:
    for inp, _ in pairs:
        if not np.array_equal(inp, apply_chain_fns(inp, fns)):
            return True
    return False


def _compile_chain_forward_map(
    pairs: list[tuple[np.ndarray, np.ndarray]], fns: tuple[GridFn, ...]
):
    """Compile chain via unique-color forward tracking (handles gravity compositions)."""
    from arc_genome.onnx.gather import build_gather_model_with_const

    shapes = {inp.shape for inp, _ in pairs}
    if len(shapes) != 1:
        return None
    ih, iw = shapes.pop()
    oh, ow = next(out.shape for _, out in pairs)
    if (oh, ow) != (ih, iw):
        return None

    idx = np.full((oh, ow, 2), -1, dtype=np.int64)
    cst = np.full((oh, ow), -1, dtype=np.int64)

    for inp0, out0 in pairs[:1]:
        for sr in range(ih):
            for sc in range(iw):
                color = int(inp0[sr, sc])
                if color == 0:
                    continue
                probe = np.zeros((ih, iw), dtype=np.int64)
                probe[sr, sc] = color
                landed = apply_chain_fns(probe, fns)
                nz = np.argwhere(landed != 0)
                if len(nz) != 1:
                    return None
                dr, dc = int(nz[0, 0]), int(nz[0, 1])
                if int(landed[dr, dc]) != color:
                    return None
                if idx[dr, dc, 0] >= 0:
                    return None
                idx[dr, dc] = [sr, sc]

    for r in range(oh):
        for c in range(ow):
            if idx[r, c, 0] < 0 and cst[r, c] < 0:
                vals = {int(out[r, c]) for _, out in pairs}
                if vals == {0}:
                    cst[r, c] = 0
                else:
                    return None

    for inp, out in pairs:
        pred = np.zeros((oh, ow), dtype=np.int64)
        for r in range(oh):
            for c in range(ow):
                if idx[r, c, 0] >= 0:
                    pred[r, c] = inp[idx[r, c, 0], idx[r, c, 1]]
                elif cst[r, c] >= 0:
                    pred[r, c] = cst[r, c]
                else:
                    return None
        if not np.array_equal(pred, out):
            return None
    return build_gather_model_with_const(ih, iw, oh, ow, idx, cst)


def _compile_chain(td, pairs: list[tuple[np.ndarray, np.ndarray]], fns: tuple[GridFn, ...]):
    """Compile matching chain to minimal gather ONNX."""
    for compiler in (
        _compile_chain_forward_map,
        _fit_gather_const_from_chain,
    ):
        m = compiler(pairs, fns)
        if m is not None:
            return m
    m = s_spatial_gather_arcgen(td) or s_position_gather_arcgen(td)
    if m is not None:
        return m
    fit = _idx_from_chain_fns(pairs, fns)
    if fit is None:
        return None
    idx, oh, ow = fit
    return build_gather_model(oh, ow, idx)


def solve_compose_arcgen(td, max_depth: int | None = None) -> object | None:
    """Search depth-2 (default) primitive chains verified on train+test+ARC-GEN."""
    cfg = get_config()
    depth = max_depth if max_depth is not None else cfg.arcgen_compose_depth
    if depth < 1:
        return None
    pairs = _pairs(td)
    if not pairs:
        return None
    ops = _build_compose_ops(pairs)
    names = list(ops.keys())

    # depth-2+ chains only (depth-1 covered by single solvers)
    for d in range(2, depth + 1):
        for combo in itertools.product(names, repeat=d):
            if combo[0] == "identity":
                continue
            fns = tuple(ops[n] for n in combo)
            if not _chain_matches(pairs, fns):
                continue
            if not _chain_non_trivial(pairs, fns):
                continue
            model = _compile_chain(td, pairs, fns)
            if model is not None:
                return model

    return None


def s_compose_arcgen(td):
    """Solver entry: ARC-GEN depth-2 composition."""
    return solve_compose_arcgen(td)

"""ARC-GEN-fitted gather solvers (Phase 15) — fit indices from train+test+arc-gen."""

from __future__ import annotations

import numpy as np

from arc_genome.config import get_config
from arc_genome.data.encoding import GH, GW, get_examples
from arc_genome.onnx.bounded import compile_bbox_relative_gather
from arc_genome.onnx.gather import build_gather_model, build_gather_model_with_const


def _arcgen_examples(td) -> list:
    cfg = get_config()
    n = cfg.arcgen_fit_samples if cfg.arcgen_validation else 50
    return td.get("train", []) + td.get("test", []) + td.get("arc-gen", [])[:n]


def _fixed_output_shape(td) -> tuple[int, int] | None:
    shapes = {tuple(np.array(ex["output"]).shape) for ex in td.get("train", []) + td.get("test", [])}
    return shapes.pop() if len(shapes) == 1 else None


def _position_idx_arcgen(td) -> tuple[np.ndarray, int, int] | None:
    out_sp = _fixed_output_shape(td)
    if out_sp is None:
        return None
    oh, ow = out_sp
    exs = _arcgen_examples(td)
    idx = np.zeros((oh, ow, 2), dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            found = False
            for sr in range(GH):
                for sc in range(GW):
                    if all(
                        sr < len(ex["input"])
                        and sc < len(ex["input"][0])
                        and r < len(ex["output"])
                        and c < len(ex["output"][0])
                        and int(ex["output"][r][c]) == int(ex["input"][sr][sc])
                        for ex in exs
                    ):
                        idx[r, c] = [sr, sc]
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
    for ex in exs:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = np.zeros_like(out)
        for r in range(min(oh, out.shape[0])):
            for c in range(min(ow, out.shape[1])):
                sr, sc = idx[r, c]
                if sr < inp.shape[0] and sc < inp.shape[1]:
                    pred[r, c] = inp[sr, sc]
        if not np.array_equal(pred, out):
            return None
    return idx, oh, ow


def _spatial_idx_arcgen(td) -> tuple[np.ndarray, np.ndarray, int, int, int, int] | None:
    from arc_genome.data.encoding import fixed_shapes
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    exs = _arcgen_examples(td)
    idx = np.full((oh, ow, 2), -1, dtype=np.int64)
    cst = np.full((oh, ow), -1, dtype=np.int64)
    for oi in range(oh):
        for oj in range(ow):
            vals = set()
            for ex in exs:
                if oi < len(ex["output"]) and oj < len(ex["output"][0]):
                    vals.add(int(ex["output"][oi][oj]))
            if len(vals) == 1:
                cst[oi, oj] = vals.pop()
            found = False
            for ri in range(GH):
                for rj in range(GW):
                    if all(
                        ri < len(ex["input"])
                        and rj < len(ex["input"][0])
                        and oi < len(ex["output"])
                        and oj < len(ex["output"][0])
                        and int(ex["input"][ri][rj]) == int(ex["output"][oi][oj])
                        for ex in exs
                    ):
                        idx[oi, oj] = [ri, rj]
                        found = True
                        break
                if found:
                    break
            if not found and cst[oi, oj] < 0:
                return None
    for ex in exs:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = np.zeros_like(out)
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                if r < oh and c < ow:
                    if idx[r, c, 0] >= 0:
                        sr, sc = idx[r, c]
                        if sr < inp.shape[0] and sc < inp.shape[1]:
                            pred[r, c] = inp[sr, sc]
                    elif cst[r, c] >= 0:
                        pred[r, c] = cst[r, c]
        if not np.array_equal(pred, out):
            return None
    return idx, cst, ih, iw, oh, ow


def s_position_gather_arcgen(td):
    """Position gather with index map fitted on train+test+ARC-GEN."""
    fit = _position_idx_arcgen(td)
    if fit is None:
        return None
    idx, oh, ow = fit
    return build_gather_model(oh, ow, idx)


def s_spatial_gather_arcgen(td):
    """Spatial gather with constants fitted on train+test+ARC-GEN."""
    fit = _spatial_idx_arcgen(td)
    if fit is None:
        return None
    idx, cst, ih, iw, oh, ow = fit
    return build_gather_model_with_const(ih, iw, oh, ow, idx, cst)


def _content_bbox(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = grid != 0
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    br, bc = int(rows[0]), int(cols[0])
    gh = int(rows[-1]) - br + 1
    gw = int(cols[-1]) - bc + 1
    return br, bc, gh, gw


def _bbox_relative_idx_arcgen(td) -> tuple[np.ndarray, int, int] | None:
    """Fit gather offsets relative to each example's input content bbox."""
    out_sp = _fixed_output_shape(td)
    if out_sp is None:
        return None
    oh, ow = out_sp
    exs = _arcgen_examples(td)
    rel = np.zeros((oh, ow, 2), dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            found = False
            for dr in range(GH):
                for dc in range(GW):
                    ok = True
                    for ex in exs:
                        inp = np.array(ex["input"])
                        out = np.array(ex["output"])
                        if r >= out.shape[0] or c >= out.shape[1]:
                            ok = False
                            break
                        bbox = _content_bbox(inp)
                        if bbox is None:
                            ok = False
                            break
                        br, bc, gh, gw = bbox
                        if dr >= gh or dc >= gw:
                            ok = False
                            break
                        sr, sc = br + dr, bc + dc
                        if sr >= inp.shape[0] or sc >= inp.shape[1]:
                            ok = False
                            break
                        if int(inp[sr, sc]) != int(out[r][c]):
                            ok = False
                            break
                    if ok:
                        rel[r, c] = [dr, dc]
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
    for ex in exs:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        bbox = _content_bbox(inp)
        if bbox is None:
            return None
        br, bc, _, _ = bbox
        pred = np.zeros((oh, ow), dtype=inp.dtype)
        for r in range(oh):
            for c in range(ow):
                dr, dc = rel[r, c]
                pred[r, c] = inp[br + dr, bc + dc]
        if not np.array_equal(pred, out[:oh, :ow]):
            return None
    return rel, oh, ow


def s_bbox_gather_arcgen(td):
    """BBox-relative gather with offsets fitted on train+test+ARC-GEN."""
    fit = _bbox_relative_idx_arcgen(td)
    if fit is None:
        return None
    rel, oh, ow = fit
    return compile_bbox_relative_gather(rel, oh, ow)


ARCgen_GATHER_SOLVERS = [
    ("bbox_gather_arcgen", s_bbox_gather_arcgen),
    ("position_gather_arcgen", s_position_gather_arcgen),
    ("spatial_gather_arcgen", s_spatial_gather_arcgen),
]

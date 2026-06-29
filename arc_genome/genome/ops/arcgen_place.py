"""ARC-GEN-fitted extract/place solvers (Phase 19) — variable layout, constant structure."""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper

from arc_genome.data.encoding import GH, GW, fixed_shapes
from arc_genome.genome.ops.arcgen_gather import (
    _arcgen_examples,
    _spatial_idx_arcgen,
    s_spatial_gather_arcgen,
)
from arc_genome.onnx.gather import build_gather_model, build_gather_model_with_const
from arc_genome.onnx.model import make_model


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


def _pairs(td) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array(ex["input"]), np.array(ex["output"])) for ex in _arcgen_examples(td)]


def _fit_spatial_gather_arcgen(td) -> object | None:
    """Spatial gather with idx/cst fitted on train+test+ARC-GEN (full grid)."""
    fit = _spatial_idx_arcgen(td)
    if fit is None:
        return None
    idx, cst, ih, iw, oh, ow = fit
    return build_gather_model_with_const(ih, iw, oh, ow, idx, cst)


def s_translate_arcgen(td):
    """Constant translation dx,dy verified on train+test+ARC-GEN."""
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    pairs = _pairs(td)
    for dx in range(-12, 13):
        for dy in range(-12, 13):
            if dx == 0 and dy == 0:
                continue
            ok = True
            for inp, out in pairs:
                shifted = np.zeros_like(inp)
                for r in range(ih):
                    for c in range(iw):
                        nr, nc = r + dx, c + dy
                        if 0 <= nr < oh and 0 <= nc < ow:
                            shifted[nr, nc] = inp[r, c]
                if not np.array_equal(shifted, out):
                    ok = False
                    break
            if not ok:
                continue
            idx = np.zeros((oh, ow, 2), dtype=np.int64)
            for r in range(oh):
                for c in range(ow):
                    sr, sc = r - dx, c - dy
                    if 0 <= sr < ih and 0 <= sc < iw:
                        idx[r, c] = [sr, sc]
            return build_gather_model(oh, ow, idx)
    return None


def s_pad_embed_arcgen(td):
    """Input embedded in larger zero canvas at fixed offset (ARC-GEN verified)."""
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if oh <= ih and ow <= iw:
        return None
    pairs = _pairs(td)
    for dr in range(0, max(0, GH - ih) + 1):
        for dc in range(0, max(0, GW - iw) + 1):
            pad_top, pad_left = dr, dc
            pad_bottom = oh - ih - pad_top
            pad_right = ow - iw - pad_left
            if pad_bottom < 0 or pad_right < 0:
                continue
            if not all(
                np.array_equal(out, np.pad(inp, ((pad_top, pad_bottom), (pad_left, pad_right))))
                for inp, out in pairs
            ):
                continue
            pad_h, pad_w = GH - oh, GW - ow
            inits = [
                numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "st"),
                numpy_helper.from_array(np.array([1, 10, ih, iw], dtype=np.int64), "en"),
            ]
            nodes = [
                helper.make_node("Slice", ["input", "st", "en"], ["sl"]),
                helper.make_node(
                    "Pad",
                    ["sl"],
                    ["output"],
                    pads=[0, 0, pad_top, pad_left, 0, 0, pad_h + pad_bottom, pad_w + pad_right],
                    value=0.0,
                ),
            ]
            return make_model(nodes, inits)
    return None


def _verify_bbox_shift(pairs, dr: int, dc: int) -> bool:
    """Output content bbox = input content bbox shifted by (dr, dc), identity paste."""
    patch_size = None
    for inp, out in pairs:
        ib = _content_bbox(inp)
        ob = _content_bbox(out)
        if ib is None and ob is None:
            if not (np.all(inp == 0) and np.all(out == 0)):
                return False
            continue
        if ib is None or ob is None:
            return False
        ibr, ibc, igh, igw = ib
        obr, obc, ogh, ogw = ob
        if (obr, obc) != (ibr + dr, ibc + dc):
            return False
        if (ogh, ogw) != (igh, igw):
            return False
        if patch_size is None:
            patch_size = (igh, igw)
        elif patch_size != (igh, igw):
            return False
        for odr in range(igh):
            for odc in range(igw):
                if int(inp[ibr + odr, ibc + odc]) != int(out[obr + odr, obc + odc]):
                    return False
    return True


def s_bbox_shift_arcgen(td):
    """Content bbox translates by constant (dr, dc) with identity patch copy."""
    pairs = _pairs(td)
    if not pairs:
        return None
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    for dr in range(-15, 16):
        for dc in range(-15, 16):
            if dr == 0 and dc == 0:
                continue
            if not _verify_bbox_shift(pairs, dr, dc):
                continue
            return _fit_spatial_gather_arcgen(td) or s_spatial_gather_arcgen(td)
    return None


def _try_patch_paste(
    pairs,
    out_r: int,
    out_c: int,
    ph: int,
    pw: int,
    flip: str,
) -> bool:
    """Fixed paste slot (out_r,out_c); patch size ph×pw; flip in {none,h,v,hv}."""
    for inp, out in pairs:
        ob = _content_bbox(out)
        if ob is None:
            if np.all(out == 0):
                continue
            return False
        obr, obc, ogh, ogw = ob
        if (obr, obc, ogh, ogw) != (out_r, out_c, ph, pw):
            return False
        ib = _content_bbox(inp)
        if ib is None:
            return False
        ibr, ibc, igh, igw = ib
        if (igh, igw) != (ph, pw):
            return False
        for odr in range(ph):
            for odc in range(pw):
                idr, idc = odr, odc
                if flip == "h":
                    idc = pw - 1 - odc
                elif flip == "v":
                    idr = ph - 1 - odr
                elif flip == "hv":
                    idr, idc = ph - 1 - odr, pw - 1 - odc
                if int(out[obr + odr, obc + odc]) != int(inp[ibr + idr, ibc + idc]):
                    return False
    return True


def _patch_size_constant(pairs) -> tuple[int, int] | None:
    sizes = set()
    for _, out in pairs:
        b = _content_bbox(out)
        if b is None:
            if np.all(out == 0):
                continue
            return None
        sizes.add((b[2], b[3]))
    return sizes.pop() if len(sizes) == 1 else None


def s_patch_paste_arcgen(td):
    """Extract patch, optional flip, paste at fixed grid slot (ARC-GEN verified)."""
    pairs = _pairs(td)
    if not pairs:
        return None
    psize = _patch_size_constant(pairs)
    if psize is None:
        return None
    ph, pw = psize
    for out_r in range(GH - ph + 1):
        for out_c in range(GW - pw + 1):
            for flip in ("none", "h", "v", "hv"):
                if not _try_patch_paste(pairs, out_r, out_c, ph, pw, flip):
                    continue
                return _fit_spatial_gather_arcgen(td) or s_spatial_gather_arcgen(td)
    return None


def s_scale_down_arcgen(td):
    """Downscale by integer factors verified on ARC-GEN."""
    pairs = _pairs(td)
    in_shapes = {inp.shape for inp, _ in pairs}
    if len(in_shapes) != 1:
        return None
    ih, iw = in_shapes.pop()
    for sh in range(2, 6):
        for sw in range(2, 6):
            oh, ow = ih // sh, iw // sw
            if oh < 1 or ow < 1 or oh * sh != ih or ow * sw != iw:
                continue
            if oh > GH or ow > GW:
                continue
            if not all(np.array_equal(out, inp[::sh, ::sw]) for inp, out in pairs):
                continue
            idx = np.zeros((oh, ow, 2), dtype=np.int64)
            for r in range(oh):
                for c in range(ow):
                    idx[r, c] = [r * sh, c * sw]
            return build_gather_model(oh, ow, idx)
    return None


def s_varying_origin_identity_arcgen(td):
    """Patch moves on grid; identity copy within content bbox (ARC-GEN verified)."""
    pairs = _pairs(td)
    if not pairs:
        return None
    psize = _patch_size_constant(pairs)
    if psize is None:
        return None
    ph, pw = psize
    for inp, out in pairs:
        ob = _content_bbox(out)
        ib = _content_bbox(inp)
        if ob is None and ib is None:
            if np.all(inp == 0) and np.all(out == 0):
                continue
            return None
        if ob is None or ib is None:
            return None
        obr, obc, ogh, ogw = ob
        ibr, ibc, igh, igw = ib
        if (ogh, ogw) != (ph, pw) or (igh, igw) != (ph, pw):
            return None
        for odr in range(ph):
            for odc in range(pw):
                if int(out[obr + odr, obc + odc]) != int(inp[ibr + odr, ibc + odc]):
                    return None
    return _fit_spatial_gather_arcgen(td) or s_spatial_gather_arcgen(td)


ARCgen_PLACE_SOLVERS = [
    ("varying_origin_paste_arcgen", s_varying_origin_identity_arcgen),
    ("bbox_shift_arcgen", s_bbox_shift_arcgen),
    ("patch_paste_arcgen", s_patch_paste_arcgen),
    ("translate_arcgen", s_translate_arcgen),
    ("pad_embed_arcgen", s_pad_embed_arcgen),
    ("scale_down_arcgen", s_scale_down_arcgen),
]

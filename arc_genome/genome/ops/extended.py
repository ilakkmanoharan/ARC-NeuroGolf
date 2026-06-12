"""Phase 3 extended analytical solvers."""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper

from arc_genome.data.encoding import GH, GW, fixed_shapes, get_examples
from arc_genome.onnx.gather import build_gather_model
from arc_genome.onnx.model import make_model


def s_translate(td):
    exs = get_examples(td)
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            if dx == 0 and dy == 0:
                continue
            ok = True
            for inp, out in exs:
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


def s_scale_down(td):
    exs = get_examples(td)
    in_shapes = {inp.shape for inp, _ in exs}
    if len(in_shapes) != 1:
        return None
    ih, iw = in_shapes.pop()
    for sh in range(2, 6):
        for sw in range(2, 6):
            oh, ow = ih // sh, iw // sw
            if oh < 1 or ow < 1 or oh * sh != ih or ow * sw != iw:
                continue
            if oh > 30 or ow > 30:
                continue
            if not all(
                np.array_equal(out, inp[::sh, ::sw]) for inp, out in exs
            ):
                continue
            idx = np.zeros((oh, ow, 2), dtype=np.int64)
            for r in range(oh):
                for c in range(ow):
                    idx[r, c] = [r * sh, c * sw]
            return build_gather_model(oh, ow, idx)
    return None


def s_pad_embed(td):
    """Output is input embedded in a larger zero canvas at fixed offset."""
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if oh <= ih and ow <= iw:
        return None
    exs = get_examples(td)
    for dr in range(0, GH - ih + 1):
        for dc in range(0, GW - iw + 1):
            if not all(
                np.array_equal(out, np.pad(inp, ((dr, oh - ih - dr), (dc, ow - iw - dc)))) for inp, out in exs
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
                    "Pad", ["sl"], ["output"],
                    pads=[0, 0, dr, dc, 0, 0, pad_h + (oh - ih - dr), pad_w + (ow - iw - dc)],
                    value=0.0,
                ),
            ]
            return make_model(nodes, inits)
    return None


def s_mask_preserve(td):
    """Output equals input where input nonzero, else fixed fill."""
    exs = get_examples(td)
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    fill_colors = set()
    for inp, out in exs:
        for r in range(ih):
            for c in range(iw):
                if inp[r, c] == 0 and out[r, c] != 0:
                    fill_colors.add(int(out[r, c]))
                if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                    return None
    if len(fill_colors) != 1:
        return None
    fill = fill_colors.pop()
    const = np.zeros((1, 10, GH, GW), dtype=np.float32)
    for r in range(ih):
        for c in range(iw):
            const[0, fill, r, c] = 1.0
    inits = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "z"),
        numpy_helper.from_array(const, "fill"),
    ]
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["mask"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["input", "z"], ["zd"]),
        helper.make_node("ReduceSum", ["zd"], ["s"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Sub", ["fill", "s"], ["bg"]),
        helper.make_node("Mul", ["bg", "mask"], ["inv"]),
        helper.make_node("Add", ["input", "inv"], ["output"]),
    ]
    return make_model(nodes, inits)


def s_mirror_complete(td):
    """Complete symmetry: output mirrors input half."""
    exs = get_examples(td)
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if oh != ih or ow != iw:
        return None
    for axis in (0, 1):
        ok = True
        for inp, out in exs:
            if axis == 1:
                half = iw // 2
                if iw % 2 != 0:
                    ok = False
                    break
                expected = np.hstack([inp[:, :half], np.fliplr(inp[:, :half])])
            else:
                half = ih // 2
                if ih % 2 != 0:
                    ok = False
                    break
                expected = np.vstack([inp[:half, :], np.flipud(inp[:half, :])])
            if not np.array_equal(out, expected):
                ok = False
                break
        if ok:
            idx = np.zeros((oh, ow, 2), dtype=np.int64)
            for r in range(oh):
                for c in range(ow):
                    if axis == 1:
                        half = iw // 2
                        sr, sc = r, c if c < half else iw - 1 - c
                    else:
                        half = ih // 2
                        sr, sc = r if r < half else ih - 1 - r, c
                    idx[r, c] = [sr, sc]
            return build_gather_model(oh, ow, idx)
    return None


EXTENDED_SOLVERS = [
    ("translate", s_translate),
    ("scale_down", s_scale_down),
    ("pad_embed", s_pad_embed),
    ("mask_preserve", s_mask_preserve),
    ("mirror_complete", s_mirror_complete),
]

"""Tier-0/1 analytical solvers that emit minimal ONNX graphs."""

from __future__ import annotations

from itertools import product as iproduct

import numpy as np
from onnx import helper, numpy_helper

from arc_genome.data.encoding import GH, GW, fixed_shapes, get_examples
from arc_genome.onnx.gather import build_gather_model, build_gather_model_with_const
from arc_genome.onnx.model import make_model


def s_identity(td):
    for ex in td["train"] + td["test"]:
        if ex["input"] != ex["output"]:
            return None
    return make_model([helper.make_node("Identity", ["input"], ["output"])])


def s_color_map(td):
    cm = {}
    for ex in td["train"] + td["test"]:
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        if inp.shape != out.shape:
            return None
        for iv, ov in zip(inp.flat, out.flat):
            iv, ov = int(iv), int(ov)
            if iv in cm and cm[iv] != ov:
                return None
            cm[iv] = ov
    w = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for ic in range(10):
        w[cm.get(ic, ic), ic, 0, 0] = 1.0
    return make_model(
        [helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1])],
        [numpy_helper.from_array(w, "W")],
    )


def s_transpose(td):
    for ex in td["train"] + td["test"]:
        if not np.array_equal(np.array(ex["output"]), np.array(ex["input"]).T):
            return None
    return make_model([helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])])


def s_flip(td):
    exs = get_examples(td)
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    for axis, flip_fn in [(0, np.flipud), (1, np.fliplr)]:
        if all(np.array_equal(out, flip_fn(inp)) for inp, out in exs):
            if axis == 0:
                idx = np.arange(GH).reshape(1, 1, GH, 1).repeat(10, 1).repeat(GW, 3)
                for r in range(ih):
                    idx[0, :, r, :] = ih - 1 - r
            else:
                idx = np.arange(GW).reshape(1, 1, 1, GW).repeat(10, 1).repeat(GH, 2)
                for c in range(iw):
                    idx[0, :, :, c] = iw - 1 - c
            ax = 2 if axis == 0 else 3
            return make_model(
                [helper.make_node("GatherElements", ["input", "idx"], ["output"], axis=ax)],
                [numpy_helper.from_array(idx.astype(np.int64), "idx")],
            )
    return None


def s_rotate(td):
    exs = get_examples(td)
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    for k in [1, 2, 3]:
        if not all(np.array_equal(out, np.rot90(inp, k)) for inp, out in exs):
            continue
        idx = np.zeros((oh, ow, 2), dtype=np.int64)
        for r in range(oh):
            for c in range(ow):
                if k == 1:
                    sr, sc = c, ih - 1 - r
                elif k == 2:
                    sr, sc = ih - 1 - r, iw - 1 - c
                else:
                    sr, sc = iw - 1 - c, r
                idx[r, c] = [sr, sc]
        return build_gather_model(oh, ow, idx)
    return None


def s_spatial_gather(td):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    exs = get_examples(td)
    idx = np.full((oh, ow, 2), -1, dtype=np.int64)
    cst = np.full((oh, ow), -1, dtype=np.int64)
    for oi in range(oh):
        for oj in range(ow):
            vals = {int(out[oi, oj]) for _, out in exs}
            if len(vals) == 1:
                cst[oi, oj] = vals.pop()
            found = False
            for ri in range(ih):
                for rj in range(iw):
                    if all(int(inp[ri, rj]) == int(out[oi, oj]) for inp, out in exs):
                        idx[oi, oj] = [ri, rj]
                        found = True
                        break
                if found:
                    break
            if not found and cst[oi, oj] < 0:
                return None
    return build_gather_model_with_const(ih, iw, oh, ow, idx, cst)


def s_tile(td):
    exs = get_examples(td)
    in_shapes = {inp.shape for inp, _ in exs}
    if len(in_shapes) != 1:
        return None
    ih, iw = in_shapes.pop()
    tiles = set()
    for inp, out in exs:
        oh, ow = out.shape
        if oh % ih or ow % iw:
            return None
        rh, rw = oh // ih, ow // iw
        if rh < 1 or rw < 1 or (rh == 1 and rw == 1):
            return None
        tiles.add((rh, rw))
    if len(tiles) != 1:
        return None
    rh, rw = tiles.pop()
    oh, ow = ih * rh, iw * rw
    if oh > 30 or ow > 30:
        return None
    for inp, out in exs:
        if not np.array_equal(out, np.tile(inp, (rh, rw))):
            return None
    pad_h, pad_w = 30 - oh, 30 - ow
    inits = [
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "st"),
        numpy_helper.from_array(np.array([1, 10, ih, iw], dtype=np.int64), "en"),
        numpy_helper.from_array(np.array([1, 1, rh, rw], dtype=np.int64), "rp"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "st", "en"], ["cr"]),
        helper.make_node("Tile", ["cr", "rp"], ["tl"]),
        helper.make_node("Pad", ["tl"], ["output"], pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], value=0.0),
    ]
    return make_model(nodes, inits)


def s_upscale(td):
    exs = get_examples(td)
    in_shapes = {inp.shape for inp, _ in exs}
    if len(in_shapes) != 1:
        return None
    ih, iw = in_shapes.pop()
    scales = set()
    for inp, out in exs:
        oh, ow = out.shape
        if oh % ih or ow % iw:
            return None
        sh, sw = oh // ih, ow // iw
        if sh < 2 or sw < 2:
            return None
        scales.add((sh, sw))
    if len(scales) != 1:
        return None
    sh, sw = scales.pop()
    oh, ow = ih * sh, iw * sw
    if oh > 30 or ow > 30:
        return None
    for inp, out in exs:
        if not np.array_equal(out, np.repeat(np.repeat(inp, sh, 0), sw, 1)):
            return None
    idx = np.zeros((oh, ow, 2), dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            idx[r, c] = [r // sh, c // sw]
    return build_gather_model(oh, ow, idx)


def s_concat(td):
    exs = get_examples(td)
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    transforms = [
        ("id", lambda x: x),
        ("fliplr", lambda x: np.fliplr(x)),
        ("flipud", lambda x: np.flipud(x)),
        ("rot180", lambda x: np.rot90(x, 2)),
    ]
    if oh == ih and ow % iw == 0 and ow > iw:
        n = ow // iw
        if 2 <= n <= 4:
            for combo in iproduct(range(4), repeat=n):
                if all(
                    np.array_equal(out, np.concatenate([transforms[t][1](inp) for t in combo], axis=1))
                    for inp, out in exs
                ):
                    idx = np.zeros((oh, ow, 2), dtype=np.int64)
                    for oi in range(oh):
                        for oj in range(ow):
                            bj = oj // iw
                            lr, lc = oi, oj % iw
                            t = transforms[combo[bj]][0]
                            if t == "id":
                                sr, sc = lr, lc
                            elif t == "fliplr":
                                sr, sc = lr, iw - 1 - lc
                            elif t == "flipud":
                                sr, sc = ih - 1 - lr, lc
                            else:
                                sr, sc = ih - 1 - lr, iw - 1 - lc
                            idx[oi, oj] = [sr, sc]
                    return build_gather_model(oh, ow, idx)
    if ow == iw and oh % ih == 0 and oh > ih:
        n = oh // ih
        if 2 <= n <= 4:
            for combo in iproduct(range(4), repeat=n):
                if all(
                    np.array_equal(out, np.concatenate([transforms[t][1](inp) for t in combo], axis=0))
                    for inp, out in exs
                ):
                    idx = np.zeros((oh, ow, 2), dtype=np.int64)
                    for oi in range(oh):
                        for oj in range(ow):
                            bi = oi // ih
                            lr, lc = oi % ih, oj
                            t = transforms[combo[bi]][0]
                            if t == "id":
                                sr, sc = lr, lc
                            elif t == "fliplr":
                                sr, sc = lr, iw - 1 - lc
                            elif t == "flipud":
                                sr, sc = ih - 1 - lr, lc
                            else:
                                sr, sc = ih - 1 - lr, iw - 1 - lc
                            idx[oi, oj] = [sr, sc]
                    return build_gather_model(oh, ow, idx)
    return None


def s_constant(td):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    exs = get_examples(td)
    outs = [out for _, out in exs]
    if not all(np.array_equal(outs[0], o) for o in outs[1:]):
        return None
    const = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(outs[0]):
        for c, v in enumerate(row):
            const[0, int(v), r, c] = 1.0
    inits = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "z"),
        numpy_helper.from_array(const, "c"),
    ]
    nodes = [
        helper.make_node("Mul", ["input", "z"], ["zd"]),
        helper.make_node("ReduceSum", ["zd"], ["s"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Add", ["s", "c"], ["output"]),
    ]
    return make_model(nodes, inits)


def s_crop(td):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if oh > ih or ow > iw:
        return None
    exs = get_examples(td)
    dr, dc = (ih - oh) // 2, (iw - ow) // 2
    for inp, out in exs:
        if not np.array_equal(out, inp[dr : dr + oh, dc : dc + ow]):
            return None
    inits = [
        numpy_helper.from_array(np.array([0, 0, dr, dc], dtype=np.int64), "st"),
        numpy_helper.from_array(np.array([1, 10, dr + oh, dc + ow], dtype=np.int64), "en"),
    ]
    pad_h, pad_w = GH - oh, GW - ow
    nodes = [
        helper.make_node("Slice", ["input", "st", "en"], ["sl"]),
        helper.make_node("Pad", ["sl"], ["output"], pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], value=0.0),
    ]
    return make_model(nodes, inits)


ANALYTICAL_SOLVERS = [
    ("identity", s_identity),
    ("constant", s_constant),
    ("color_map", s_color_map),
    ("transpose", s_transpose),
    ("flip", s_flip),
    ("rotate", s_rotate),
    ("tile", s_tile),
    ("upscale", s_upscale),
    ("concat", s_concat),
    ("spatial_gather", s_spatial_gather),
    ("crop", s_crop),
]

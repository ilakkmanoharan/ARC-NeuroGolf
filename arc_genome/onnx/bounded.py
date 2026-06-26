"""Compact dynamic bounded-flip ONNX compiler (Phase 11)."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from arc_genome.data.encoding import CH, GH, GW
from arc_genome.onnx.model import make_model

DT = TensorProto.FLOAT


def _f32(name, val):
    return numpy_helper.from_array(np.array(val, dtype=np.float32), name)


def _i64(name, val):
    return numpy_helper.from_array(np.array(val, dtype=np.int64), name)


def _bbox_scalars(nodes, inits, occ_name, tag):
    inits.append(_f32(f"{tag}_half", 0.5))
    inits.append(_f32(f"{tag}_one", 1.0))
    rows = []
    for r in range(GH):
        s, e = f"{tag}_rs{r}", f"{tag}_re{r}"
        inits += [_i64(s, [0, 0, r, 0]), _i64(e, [1, 1, r + 1, GW])]
        sl = f"{tag}_rsl{r}"
        nodes.append(helper.make_node("Slice", [occ_name, s, e], [sl]))
        mx = f"{tag}_rmx{r}"
        nodes.append(helper.make_node("ReduceMax", [sl], [mx], keepdims=0, axes=[0, 1, 2, 3]))
        gt = f"{tag}_rgt{r}"
        nodes.append(helper.make_node("Greater", [mx, f"{tag}_half"], [gt]))
        cst = f"{tag}_rc{r}"
        nodes.append(helper.make_node("Cast", [gt], [cst], to=DT))
        wf = f"{tag}_wf{r}"
        inits.append(_f32(wf, float(r)))
        tm = f"{tag}_rt{r}"
        nodes.append(helper.make_node("Mul", [cst, wf], [tm]))
        rows.append(tm)
    gh = rows[0]
    for i, tm in enumerate(rows[1:], 1):
        nm = f"{tag}_gh{i}"
        nodes.append(helper.make_node("Max", [gh, tm], [nm]))
        gh = nm
    nodes.append(helper.make_node("Add", [gh, f"{tag}_one"], [f"{tag}_ghp1"]))
    gh = f"{tag}_ghp1"
    cols = []
    for c in range(GW):
        s, e = f"{tag}_cs{c}", f"{tag}_ce{c}"
        inits += [_i64(s, [0, 0, 0, c]), _i64(e, [1, 1, GH, c + 1])]
        sl = f"{tag}_csl{c}"
        nodes.append(helper.make_node("Slice", [occ_name, s, e], [sl]))
        mx = f"{tag}_cmx{c}"
        nodes.append(helper.make_node("ReduceMax", [sl], [mx], keepdims=0, axes=[0, 1, 2, 3]))
        gt = f"{tag}_cgt{c}"
        nodes.append(helper.make_node("Greater", [mx, f"{tag}_half"], [gt]))
        cst = f"{tag}_cc{c}"
        nodes.append(helper.make_node("Cast", [gt], [cst], to=DT))
        wf = f"{tag}_cf{c}"
        inits.append(_f32(wf, float(c)))
        tm = f"{tag}_ct{c}"
        nodes.append(helper.make_node("Mul", [cst, wf], [tm]))
        cols.append(tm)
    gw = cols[0]
    for i, tm in enumerate(cols[1:], 1):
        nm = f"{tag}_gw{i}"
        nodes.append(helper.make_node("Max", [gw, tm], [nm]))
        gw = nm
    nodes.append(helper.make_node("Add", [gw, f"{tag}_one"], [f"{tag}_gwp1"]))
    gw = f"{tag}_gwp1"
    return gh, gw


def _bbox_corners_scalars(nodes, inits, occ_name, tag):
    """Tight content bbox top-left (br, bc) as dynamic scalars."""
    inits.append(_f32(f"{tag}_half", 0.5))
    inits.append(_f32(f"{tag}_one", 1.0))
    row_sentinel = f"{tag}_rsent"
    col_sentinel = f"{tag}_csent"
    inits.append(_f32(row_sentinel, float(GH)))
    inits.append(_f32(col_sentinel, float(GW)))

    row_weights = []
    for r in range(GH):
        s, e = f"{tag}_rs{r}", f"{tag}_re{r}"
        inits += [_i64(s, [0, 0, r, 0]), _i64(e, [1, 1, r + 1, GW])]
        sl = f"{tag}_rsl{r}"
        nodes.append(helper.make_node("Slice", [occ_name, s, e], [sl]))
        mx = f"{tag}_rmx{r}"
        nodes.append(helper.make_node("ReduceMax", [sl], [mx], keepdims=0, axes=[0, 1, 2, 3]))
        gt = f"{tag}_rgt{r}"
        nodes.append(helper.make_node("Greater", [mx, f"{tag}_half"], [gt]))
        occ_f = f"{tag}_rocc{r}"
        nodes.append(helper.make_node("Cast", [gt], [occ_f], to=DT))
        wf = f"{tag}_rwf{r}"
        inits.append(_f32(wf, float(r)))
        tm = f"{tag}_rt{r}"
        nodes.append(helper.make_node("Mul", [occ_f, wf], [tm]))
        not_occ = f"{tag}_rnocc{r}"
        nodes.append(helper.make_node("Sub", [f"{tag}_one", occ_f], [not_occ]))
        sent = f"{tag}_rsnt{r}"
        nodes.append(helper.make_node("Mul", [not_occ, row_sentinel], [sent]))
        wt = f"{tag}_rwt{r}"
        nodes.append(helper.make_node("Add", [tm, sent], [wt]))
        row_weights.append(wt)

    br = row_weights[0]
    for i, wt in enumerate(row_weights[1:], 1):
        nm = f"{tag}_brmin{i}"
        nodes.append(helper.make_node("Min", [br, wt], [nm]))
        br = nm

    col_weights = []
    for c in range(GW):
        s, e = f"{tag}_cs{c}", f"{tag}_ce{c}"
        inits += [_i64(s, [0, 0, 0, c]), _i64(e, [1, 1, GH, c + 1])]
        sl = f"{tag}_csl{c}"
        nodes.append(helper.make_node("Slice", [occ_name, s, e], [sl]))
        mx = f"{tag}_cmx{c}"
        nodes.append(helper.make_node("ReduceMax", [sl], [mx], keepdims=0, axes=[0, 1, 2, 3]))
        gt = f"{tag}_cgt{c}"
        nodes.append(helper.make_node("Greater", [mx, f"{tag}_half"], [gt]))
        occ_f = f"{tag}_cocc{c}"
        nodes.append(helper.make_node("Cast", [gt], [occ_f], to=DT))
        wf = f"{tag}_cwf{c}"
        inits.append(_f32(wf, float(c)))
        tm = f"{tag}_ct{c}"
        nodes.append(helper.make_node("Mul", [occ_f, wf], [tm]))
        not_occ = f"{tag}_cnocc{c}"
        nodes.append(helper.make_node("Sub", [f"{tag}_one", occ_f], [not_occ]))
        sent = f"{tag}_csnt{c}"
        nodes.append(helper.make_node("Mul", [not_occ, col_sentinel], [sent]))
        wt = f"{tag}_cwt{c}"
        nodes.append(helper.make_node("Add", [tm, sent], [wt]))
        col_weights.append(wt)

    bc = col_weights[0]
    for i, wt in enumerate(col_weights[1:], 1):
        nm = f"{tag}_bcmin{i}"
        nodes.append(helper.make_node("Min", [bc, wt], [nm]))
        bc = nm

    return br, bc


def _content_occ_nodes(nodes, inits, input_name: str, occ_name: str):
    """Occupancy map excluding background channel 0 (matches tight content bbox)."""
    inits.append(_i64("ch1s", [0, 1, 0, 0]))
    inits.append(_i64("ch1e", [1, CH, GH, GW]))
    nodes.append(helper.make_node("Slice", [input_name, "ch1s", "ch1e"], ["no_bg"]))
    nodes.append(helper.make_node("ReduceMax", ["no_bg"], [occ_name], keepdims=1, axes=[1]))


def compile_bbox_relative_gather(rel_idx: np.ndarray, oh: int, ow: int):
    """Dynamic gather: out[r,c] = input[br+dr, bc+dc] with fitted relative offsets."""
    nodes = []
    inits = [
        _f32("one", 1.0),
        _i64("flat_shape", [1, CH, GH * GW]),
        _i64("out_shape", [1, CH, GH, GW]),
        _i64("mask_shape", [1, 1, GH, GW]),
        _i64("sh1", [1]),
        _f32("gw_f", float(GW)),
    ]
    for r in range(oh):
        for c in range(ow):
            dr, dc = int(rel_idx[r, c, 0]), int(rel_idx[r, c, 1])
            inits.append(_f32(f"dr{r}c{c}", float(dr)))
            inits.append(_f32(f"dc{r}c{c}", float(dc)))

    _content_occ_nodes(nodes, inits, "input", "occ")
    br, bc = _bbox_corners_scalars(nodes, inits, "occ", "bb")

    flat_indices = []
    masks = []
    for r in range(GH):
        for c in range(GW):
            tag = f"r{r}c{c}"
            if r < oh and c < ow:
                nodes.append(helper.make_node("Add", [br, f"dr{r}c{c}"], [f"sr_{tag}"]))
                nodes.append(helper.make_node("Add", [bc, f"dc{r}c{c}"], [f"sc_{tag}"]))
                nodes.append(helper.make_node("Mul", [f"sr_{tag}", "gw_f"], [f"rowoff_{tag}"]))
                nodes.append(helper.make_node("Add", [f"rowoff_{tag}", f"sc_{tag}"], [f"flatf_{tag}"]))
                inits.append(_f32(f"omsk_{tag}", 1.0))
                nodes.append(helper.make_node("Reshape", [f"omsk_{tag}", "sh1"], [f"msk1_{tag}"]))
            else:
                inits.append(_f32(f"flatf_{tag}", 0.0))
                inits.append(_f32(f"omsk_{tag}", 0.0))
                nodes.append(helper.make_node("Reshape", [f"omsk_{tag}", "sh1"], [f"msk1_{tag}"]))
            fi = f"fi_{tag}"
            nodes.append(helper.make_node("Cast", [f"flatf_{tag}"], [fi], to=TensorProto.INT64))
            fi1 = f"fi1_{tag}"
            nodes.append(helper.make_node("Reshape", [fi, "sh1"], [fi1]))
            flat_indices.append(fi1)
            masks.append(f"msk1_{tag}")

    nodes.append(helper.make_node("Concat", flat_indices, ["flat_idx"], axis=0))
    nodes.append(helper.make_node("Concat", masks, ["flat_mask"], axis=0))
    nodes.append(helper.make_node("Reshape", ["flat_mask", "mask_shape"], ["mask2d"]))
    nodes.append(helper.make_node("Reshape", ["input", "flat_shape"], ["flat_in"]))
    nodes.append(helper.make_node("Gather", ["flat_in", "flat_idx"], ["gathered"], axis=2))
    nodes.append(helper.make_node("Reshape", ["gathered", "out_shape"], ["raw_out"]))
    nodes.append(helper.make_node("Mul", ["raw_out", "mask2d"], ["output"]))
    return make_model(nodes, inits)


def compile_dynamic_upscale2():
    """2× nearest-neighbor upscale within runtime content bbox."""
    return _compile_dynamic_upscale2_slim()


def compile_dynamic_flip(axis: str, *, slim: bool = True):
    if axis not in ("h", "v"):
        return None
    if slim:
        return _compile_dynamic_flip_slim(axis)
    return _compile_dynamic_flip_full(axis)


def _compile_dynamic_flip_full(axis: str):
    if axis not in ("h", "v"):
        return None
    nodes = []
    inits = [
        _f32("one", 1.0),
        _i64("flat_shape", [1, GH * GW]),
        _i64("out_shape", [1, 1, GH, GW]),
        _f32("gw_f", float(GW)),
    ]
    for r in range(GH):
        inits.append(_f32(f"rf{r}", float(r)))
    for c in range(GW):
        inits.append(_f32(f"cf{c}", float(c)))

    nodes.append(helper.make_node("ReduceMax", ["input"], ["occ"], keepdims=1, axes=[1]))
    gh, gw = _bbox_scalars(nodes, inits, "occ", "bb")

    flat_indices = []
    masks = []
    for r in range(GH):
        for c in range(GW):
            tag = f"r{r}c{c}"
            rf, cf = f"rf{r}", f"cf{c}"

            if axis == "h":
                nodes.append(helper.make_node("Sub", [gh, "one"], [f"ghm1_{tag}"]))
                nodes.append(helper.make_node("Sub", [f"ghm1_{tag}", rf], [f"sr_{tag}"]))
                src = f"sr_{tag}"
                nodes.append(helper.make_node("Mul", [src, "gw_f"], [f"rowoff_{tag}"]))
                nodes.append(helper.make_node("Add", [f"rowoff_{tag}", cf], [f"flatf_{tag}"]))
            else:
                nodes.append(helper.make_node("Sub", [gw, "one"], [f"gwm1_{tag}"]))
                nodes.append(helper.make_node("Sub", [f"gwm1_{tag}", cf], [f"sc_{tag}"]))
                nodes.append(helper.make_node("Mul", [rf, "gw_f"], [f"rowoff_{tag}"]))
                nodes.append(helper.make_node("Add", [f"rowoff_{tag}", f"sc_{tag}"], [f"flatf_{tag}"]))

            fi = f"fi_{tag}"
            nodes.append(helper.make_node("Cast", [f"flatf_{tag}"], [fi], to=TensorProto.INT64))
            fi1 = f"fi1_{tag}"
            inits.append(_i64(f"sh1_{tag}", [1]))
            nodes.append(helper.make_node("Reshape", [fi, f"sh1_{tag}"], [fi1]))
            flat_indices.append(fi1)

            nodes.append(helper.make_node("Greater", [gh, rf], [f"rgt_{tag}"]))
            nodes.append(helper.make_node("Greater", [gw, cf], [f"cgt_{tag}"]))
            nodes.append(helper.make_node("Cast", [f"rgt_{tag}"], [f"rm_{tag}"], to=DT))
            nodes.append(helper.make_node("Cast", [f"cgt_{tag}"], [f"cm_{tag}"], to=DT))
            msk = f"msk_{tag}"
            nodes.append(helper.make_node("Mul", [f"rm_{tag}", f"cm_{tag}"], [msk]))
            msk1 = f"msk1_{tag}"
            nodes.append(helper.make_node("Reshape", [msk, f"sh1_{tag}"], [msk1]))
            masks.append(msk1)

    nodes.append(helper.make_node("Concat", flat_indices, ["flat_idx"], axis=0))
    nodes.append(helper.make_node("Concat", masks, ["flat_mask"], axis=0))
    nodes.append(helper.make_node("Reshape", ["flat_mask", "out_shape"], ["mask2d"]))

    ch_outs = []
    for k in range(CH):
        ks, ke = f"ks{k}", f"ke{k}"
        inits += [_i64(ks, [0, k, 0, 0]), _i64(ke, [1, k + 1, GH, GW])]
        xk = f"xk{k}"
        nodes.append(helper.make_node("Slice", ["input", ks, ke], [xk]))
        flat = f"flat{k}"
        nodes.append(helper.make_node("Reshape", [xk, "flat_shape"], [flat]))
        g = f"g{k}"
        nodes.append(helper.make_node("Gather", [flat, "flat_idx"], [g], axis=1))
        resh = f"rs{k}"
        nodes.append(helper.make_node("Reshape", [g, "out_shape"], [resh]))
        masked = f"msk{k}"
        nodes.append(helper.make_node("Mul", [resh, "mask2d"], [masked]))
        ch_outs.append(masked)

    nodes.append(helper.make_node("Concat", ch_outs, ["output"], axis=1))
    return make_model(nodes, inits)


def _compile_dynamic_flip_slim(axis: str):
    """Single Gather over spatial axis for all channels — ~10× smaller graph."""
    nodes = []
    inits = [
        _f32("one", 1.0),
        _i64("flat_shape", [1, CH, GH * GW]),
        _i64("out_shape", [1, CH, GH, GW]),
        _i64("mask_shape", [1, 1, GH, GW]),
        _i64("sh1", [1]),
        _f32("gw_f", float(GW)),
    ]
    for r in range(GH):
        inits.append(_f32(f"rf{r}", float(r)))
    for c in range(GW):
        inits.append(_f32(f"cf{c}", float(c)))

    nodes.append(helper.make_node("ReduceMax", ["input"], ["occ"], keepdims=1, axes=[1]))
    gh, gw = _bbox_scalars(nodes, inits, "occ", "bb")

    flat_indices = []
    masks = []
    for r in range(GH):
        for c in range(GW):
            tag = f"r{r}c{c}"
            rf, cf = f"rf{r}", f"cf{c}"
            if axis == "h":
                nodes.append(helper.make_node("Sub", [gh, "one"], [f"ghm1_{tag}"]))
                nodes.append(helper.make_node("Sub", [f"ghm1_{tag}", rf], [f"sr_{tag}"]))
                nodes.append(helper.make_node("Mul", [f"sr_{tag}", "gw_f"], [f"rowoff_{tag}"]))
                nodes.append(helper.make_node("Add", [f"rowoff_{tag}", cf], [f"flatf_{tag}"]))
            else:
                nodes.append(helper.make_node("Sub", [gw, "one"], [f"gwm1_{tag}"]))
                nodes.append(helper.make_node("Sub", [f"gwm1_{tag}", cf], [f"sc_{tag}"]))
                nodes.append(helper.make_node("Mul", [rf, "gw_f"], [f"rowoff_{tag}"]))
                nodes.append(helper.make_node("Add", [f"rowoff_{tag}", f"sc_{tag}"], [f"flatf_{tag}"]))
            fi = f"fi_{tag}"
            nodes.append(helper.make_node("Cast", [f"flatf_{tag}"], [fi], to=TensorProto.INT64))
            fi1 = f"fi1_{tag}"
            nodes.append(helper.make_node("Reshape", [fi, "sh1"], [fi1]))
            flat_indices.append(fi1)
            nodes.append(helper.make_node("Greater", [gh, rf], [f"rgt_{tag}"]))
            nodes.append(helper.make_node("Greater", [gw, cf], [f"cgt_{tag}"]))
            nodes.append(helper.make_node("Cast", [f"rgt_{tag}"], [f"rm_{tag}"], to=DT))
            nodes.append(helper.make_node("Cast", [f"cgt_{tag}"], [f"cm_{tag}"], to=DT))
            msk = f"msk_{tag}"
            nodes.append(helper.make_node("Mul", [f"rm_{tag}", f"cm_{tag}"], [msk]))
            msk1 = f"msk1_{tag}"
            nodes.append(helper.make_node("Reshape", [msk, "sh1"], [msk1]))
            masks.append(msk1)

    nodes.append(helper.make_node("Concat", flat_indices, ["flat_idx"], axis=0))
    nodes.append(helper.make_node("Concat", masks, ["flat_mask"], axis=0))
    nodes.append(helper.make_node("Reshape", ["flat_mask", "mask_shape"], ["mask2d"]))

    nodes.append(helper.make_node("Reshape", ["input", "flat_shape"], ["flat_in"]))
    nodes.append(helper.make_node("Gather", ["flat_in", "flat_idx"], ["gathered"], axis=2))
    nodes.append(helper.make_node("Reshape", ["gathered", "out_shape"], ["raw_out"]))
    nodes.append(helper.make_node("Mul", ["raw_out", "mask2d"], ["output"]))
    return make_model(nodes, inits)


def _compile_dynamic_upscale2_slim():
    """Dynamic 2× upscale: out[r,c] = in[r//2, c//2] within 2×gh × 2×gw bbox."""
    nodes = []
    inits = [
        _f32("one", 1.0),
        _f32("two", 2.0),
        _i64("flat_shape", [1, CH, GH * GW]),
        _i64("out_shape", [1, CH, GH, GW]),
        _i64("mask_shape", [1, 1, GH, GW]),
        _i64("sh1", [1]),
        _f32("gw_f", float(GW)),
    ]
    for r in range(GH):
        inits.append(_f32(f"rf{r}", float(r // 2)))
        inits.append(_f32(f"or{r}", float(r)))
    for c in range(GW):
        inits.append(_f32(f"cf{c}", float(c // 2)))
        inits.append(_f32(f"oc{c}", float(c)))

    nodes.append(helper.make_node("ReduceMax", ["input"], ["occ"], keepdims=1, axes=[1]))
    gh, gw = _bbox_scalars(nodes, inits, "occ", "bb")
    nodes.append(helper.make_node("Mul", [gh, "two"], ["gh2"]))
    nodes.append(helper.make_node("Mul", [gw, "two"], ["gw2"]))

    flat_indices = []
    masks = []
    for r in range(GH):
        for c in range(GW):
            tag = f"r{r}c{c}"
            rf, cf = f"rf{r}", f"cf{c}"
            nodes.append(helper.make_node("Mul", [rf, "gw_f"], [f"rowoff_{tag}"]))
            nodes.append(helper.make_node("Add", [f"rowoff_{tag}", cf], [f"flatf_{tag}"]))
            fi = f"fi_{tag}"
            nodes.append(helper.make_node("Cast", [f"flatf_{tag}"], [fi], to=TensorProto.INT64))
            fi1 = f"fi1_{tag}"
            nodes.append(helper.make_node("Reshape", [fi, "sh1"], [fi1]))
            flat_indices.append(fi1)
            nodes.append(helper.make_node("Greater", ["gh2", f"or{r}"], [f"rgt_{tag}"]))
            nodes.append(helper.make_node("Greater", ["gw2", f"oc{c}"], [f"cgt_{tag}"]))
            nodes.append(helper.make_node("Cast", [f"rgt_{tag}"], [f"rm_{tag}"], to=DT))
            nodes.append(helper.make_node("Cast", [f"cgt_{tag}"], [f"cm_{tag}"], to=DT))
            msk = f"msk_{tag}"
            nodes.append(helper.make_node("Mul", [f"rm_{tag}", f"cm_{tag}"], [msk]))
            msk1 = f"msk1_{tag}"
            nodes.append(helper.make_node("Reshape", [msk, "sh1"], [msk1]))
            masks.append(msk1)

    nodes.append(helper.make_node("Concat", flat_indices, ["flat_idx"], axis=0))
    nodes.append(helper.make_node("Concat", masks, ["flat_mask"], axis=0))
    nodes.append(helper.make_node("Reshape", ["flat_mask", "mask_shape"], ["mask2d"]))
    nodes.append(helper.make_node("Reshape", ["input", "flat_shape"], ["flat_in"]))
    nodes.append(helper.make_node("Gather", ["flat_in", "flat_idx"], ["gathered"], axis=2))
    nodes.append(helper.make_node("Reshape", ["gathered", "out_shape"], ["raw_out"]))
    nodes.append(helper.make_node("Mul", ["raw_out", "mask2d"], ["output"]))
    return make_model(nodes, inits)

"""Convolution least-squares fallback solvers."""

from __future__ import annotations

import time

import numpy as np
from onnx import helper, numpy_helper

from arc_genome.config import get_config
from arc_genome.data.encoding import GH, GW, fixed_shapes, get_examples


def _fitting_examples(task_data: dict):
    """Train+test plus ARC-GEN pairs with matching shapes (up to 50)."""
    exs = get_examples(task_data)
    if not exs:
        return exs
    in_shapes = {inp.shape for inp, _ in exs}
    out_shapes = {out.shape for _, out in exs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return exs
    in_shape, out_shape = list(in_shapes)[0], list(out_shapes)[0]
    extra = []
    for ex in task_data.get("arc-gen", []):
        inp = np.array(ex["input"], dtype=np.int64)
        out = np.array(ex["output"], dtype=np.int64)
        if inp.shape == in_shape and out.shape == out_shape:
            extra.append((inp, out))
    return exs + extra[:50]


from arc_genome.onnx.model import make_model, save_model, validate_model


def _kernel_sizes(max_k: int | None = None) -> list[int]:
    cfg = get_config()
    cap = max_k if max_k is not None else cfg.max_kernel
    return list(range(1, cap + 1, 2))


def _sparsify_weights(wconv: np.ndarray) -> tuple[np.ndarray, int, int]:
    kh, kw = wconv.shape[2], wconv.shape[3]
    if not get_config().conv_sparsify:
        return wconv, kh, kw
    w = wconv.copy()
    w[np.abs(w) < 1e-6] = 0.0
    nz = np.where(np.abs(w) > 0)
    if len(nz[0]) == 0:
        return w, kh, kw
    r0, r1 = nz[2].min(), nz[2].max() + 1
    c0, c1 = nz[3].min(), nz[3].max() + 1
    w_small = w[:, :, r0:r1, c0:c1]
    return w_small, w_small.shape[2], w_small.shape[3]


def _lstsq_conv(exs_raw, ks, use_bias, use_full_30=False):
    pad = ks // 2
    feat = 10 * ks * ks + (1 if use_bias else 0)
    if feat > 20000:
        return None

    patches, targets = [], []
    for inp_g, out_g in exs_raw:
        ih, iw = inp_g.shape
        if use_full_30:
            oh_full = np.zeros((10, GH, GW), dtype=np.float64)
            for c in range(10):
                oh_full[c, :ih, :iw] = inp_g == c
            oh_pad = np.pad(oh_full, ((0, 0), (pad, pad), (pad, pad)))
        else:
            oh_enc = np.zeros((10, ih, iw), dtype=np.float64)
            for c in range(10):
                oh_enc[c] = inp_g == c
            oh_pad = np.pad(oh_enc, ((0, 0), (pad, pad), (pad, pad)))

        oh, ow = out_g.shape
        for r in range(oh):
            for c in range(ow):
                p = oh_pad[:, r : r + ks, c : c + ks].flatten()
                if use_bias:
                    p = np.append(p, 1.0)
                patches.append(p)
                targets.append(int(out_g[r, c]))

    if feat > 5000 and len(patches) > 2000:
        return None

    p_mat = np.array(patches, dtype=np.float64)
    t = np.array(targets, dtype=np.int64)
    t_oh = np.zeros((len(t), 10), dtype=np.float64)
    for i, tv in enumerate(t):
        t_oh[i, tv] = 1.0

    wt = np.linalg.lstsq(p_mat, t_oh, rcond=None)[0]
    if not np.array_equal(np.argmax(p_mat @ wt, axis=1), t):
        return None

    if use_bias:
        wconv = wt[:-1].T.reshape(10, 10, ks, ks).astype(np.float32)
        b = wt[-1].astype(np.float32)
    else:
        wconv = wt.T.reshape(10, 10, ks, ks).astype(np.float32)
        b = None
    wconv, kh, kw = _sparsify_weights(wconv)
    return wconv, b, kh, kw


def _emit_conv_fixed(td, path, exs, ih, iw, wconv, b, kh, kw):
    conv_pad_h, conv_pad_w = kh // 2, kw // 2
    inits = [
        numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "sl_st"),
        numpy_helper.from_array(np.array([1, 10, ih, iw], dtype=np.int64), "sl_en"),
        numpy_helper.from_array(wconv, "W"),
        numpy_helper.from_array(np.array(10, dtype=np.int64), "depth"),
        numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), "ohvals"),
    ]
    conv_inputs = ["grid", "W"]
    if b is not None:
        inits.append(numpy_helper.from_array(b, "B"))
        conv_inputs.append("B")
    nodes = [
        helper.make_node("Slice", ["input", "sl_st", "sl_en"], ["grid"]),
        helper.make_node("Conv", conv_inputs, ["co"], kernel_shape=[kh, kw], pads=[conv_pad_h, conv_pad_w, conv_pad_h, conv_pad_w]),
        helper.make_node("ArgMax", ["co"], ["am"], axis=1, keepdims=0),
        helper.make_node("OneHot", ["am", "depth", "ohvals"], ["oh_out"], axis=1),
    ]
    out_pad_h, out_pad_w = GH - ih, GW - iw
    if out_pad_h > 0 or out_pad_w > 0:
        nodes.append(
            helper.make_node("Pad", ["oh_out"], ["output"], pads=[0, 0, 0, 0, 0, 0, out_pad_h, out_pad_w], value=0.0)
        )
    else:
        nodes.append(helper.make_node("Identity", ["oh_out"], ["output"]))
    model = make_model(nodes, inits)
    save_model(model, path)
    return model if validate_model(path, td) else None


def solve_conv_fixed(td, path, time_budget=30.0, max_kernel: int | None = None):
    exs = _fitting_examples(td)
    if any(inp.shape != out.shape for inp, out in exs):
        return None
    shapes = {inp.shape for inp, _ in exs}
    if len(shapes) != 1:
        return None
    ih, iw = shapes.pop()

    t_start = time.time()
    for use_bias in [False, True]:
        for ks in _kernel_sizes(max_kernel):
            if time.time() - t_start > time_budget:
                return None
            result = _lstsq_conv(exs, ks, use_bias, use_full_30=False)
            if result is None:
                continue
            wconv, b, kh, kw = result
            model = _emit_conv_fixed(td, path, exs, ih, iw, wconv, b, kh, kw)
            if model is not None:
                return model
    return None


def solve_conv_variable(td, path, time_budget=30.0, max_kernel: int | None = None):
    exs = _fitting_examples(td)
    if any(inp.shape != out.shape for inp, out in exs):
        return None

    t_start = time.time()
    for use_bias in [False, True]:
        for ks in _kernel_sizes(max_kernel):
            if time.time() - t_start > time_budget:
                return None
            result = _lstsq_conv(exs, ks, use_bias, use_full_30=True)
            if result is None:
                continue
            wconv, b, kh, kw = result
            pad_h, pad_w = kh // 2, kw // 2
            inits = [
                numpy_helper.from_array(wconv, "W"),
                numpy_helper.from_array(np.array(10, dtype=np.int64), "depth"),
                numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), "ohvals"),
            ]
            conv_inputs = ["input", "W"]
            if b is not None:
                inits.append(numpy_helper.from_array(b, "B"))
                conv_inputs.append("B")
            nodes = [
                helper.make_node("ReduceSum", ["input"], ["mask"], axes=[1], keepdims=1),
                helper.make_node("Conv", conv_inputs, ["co"], kernel_shape=[kh, kw], pads=[pad_h, pad_w, pad_h, pad_w]),
                helper.make_node("ArgMax", ["co"], ["am"], axis=1, keepdims=0),
                helper.make_node("OneHot", ["am", "depth", "ohvals"], ["oh_out"], axis=1),
                helper.make_node("Mul", ["oh_out", "mask"], ["output"]),
            ]
            model = make_model(nodes, inits)
            save_model(model, path)
            if validate_model(path, td):
                return model
    return None


def solve_conv_diffshape(td, path, time_budget=30.0, max_kernel: int | None = None):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if ih == oh and iw == ow or oh > ih or ow > iw or oh > 30 or ow > 30:
        return None

    exs = _fitting_examples(td)
    t_start = time.time()
    cap = max_kernel or get_config().max_kernel

    for dr_off, dc_off in [(0, 0), ((ih - oh) // 2, (iw - ow) // 2)]:
        for use_bias in [False, True]:
            for ks in _kernel_sizes(cap):
                if time.time() - t_start > time_budget:
                    return None
                pad = ks // 2
                feat = 10 * ks * ks + (1 if use_bias else 0)
                if feat > 10000:
                    continue

                patches, targets = [], []
                valid = True
                for inp_g, out_g in exs:
                    oh_enc = np.zeros((10, ih, iw), dtype=np.float64)
                    for c in range(10):
                        oh_enc[c] = inp_g == c
                    oh_pad = np.pad(oh_enc, ((0, 0), (pad, pad), (pad, pad)))
                    for r in range(oh):
                        for c in range(ow):
                            sr, sc = r + dr_off, c + dc_off
                            if sr < 0 or sr >= ih or sc < 0 or sc >= iw:
                                valid = False
                                break
                            p = oh_pad[:, sr : sr + ks, sc : sc + ks].flatten()
                            if use_bias:
                                p = np.append(p, 1.0)
                            patches.append(p)
                            targets.append(int(out_g[r, c]))
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid or (feat > 5000 and len(patches) > 2000):
                    continue

                p_mat = np.array(patches, dtype=np.float64)
                t = np.array(targets, dtype=np.int64)
                t_oh = np.zeros((len(t), 10), dtype=np.float64)
                for i, tv in enumerate(t):
                    t_oh[i, tv] = 1.0

                wt = np.linalg.lstsq(p_mat, t_oh, rcond=None)[0]
                if not np.array_equal(np.argmax(p_mat @ wt, axis=1), t):
                    continue

                if use_bias:
                    wconv = wt[:-1].T.reshape(10, 10, ks, ks).astype(np.float32)
                    b = wt[-1].astype(np.float32)
                else:
                    wconv = wt.T.reshape(10, 10, ks, ks).astype(np.float32)
                    b = None
                wconv, kh, kw = _sparsify_weights(wconv)
                pad_h, pad_w = kh // 2, kw // 2

                out_pad_h, out_pad_w = GH - oh, GW - ow
                inits = [
                    numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), "sl_st"),
                    numpy_helper.from_array(np.array([1, 10, ih, iw], dtype=np.int64), "sl_en"),
                    numpy_helper.from_array(wconv, "W"),
                    numpy_helper.from_array(np.array(10, dtype=np.int64), "depth"),
                    numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), "ohvals"),
                    numpy_helper.from_array(np.array([0, 0, dr_off, dc_off], dtype=np.int64), "cr_st"),
                    numpy_helper.from_array(np.array([1, 10, dr_off + oh, dc_off + ow], dtype=np.int64), "cr_en"),
                ]
                conv_inputs = ["grid", "W"]
                if b is not None:
                    inits.append(numpy_helper.from_array(b, "B"))
                    conv_inputs.append("B")

                nodes = [
                    helper.make_node("Slice", ["input", "sl_st", "sl_en"], ["grid"]),
                    helper.make_node("Conv", conv_inputs, ["co"], kernel_shape=[kh, kw], pads=[pad_h, pad_w, pad_h, pad_w]),
                    helper.make_node("Slice", ["co", "cr_st", "cr_en"], ["co_crop"]),
                    helper.make_node("ArgMax", ["co_crop"], ["am"], axis=1, keepdims=0),
                    helper.make_node("OneHot", ["am", "depth", "ohvals"], ["oh_out"], axis=1),
                    helper.make_node("Pad", ["oh_out"], ["output"], pads=[0, 0, 0, 0, 0, 0, out_pad_h, out_pad_w], value=0.0),
                ]
                model = make_model(nodes, inits)
                save_model(model, path)
                if validate_model(path, td):
                    return model
    return None


def solve_conv_v2(td, path, time_budget=90.0):
    """Phase 5: extended kernel search for stubborn tasks."""
    cfg = get_config()
    mk = cfg.unsolved_max_kernel
    for solver in (solve_conv_fixed, solve_conv_variable, solve_conv_diffshape):
        model = solver(td, path, time_budget=time_budget / 3, max_kernel=mk)
        if model is not None:
            return model
    return None

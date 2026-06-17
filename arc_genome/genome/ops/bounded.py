"""Phase 11 bounded world-model solvers + BLF belief routing."""

from __future__ import annotations

import numpy as np

from arc_genome.data.encoding import get_examples, to_onehot
from arc_genome.onnx.bounded import compile_dynamic_flip, compile_dynamic_upscale2


def _verify_numpy(fn, td) -> bool:
    for ex in td["train"] + td["test"]:
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        pred = fn(inp)
        if not np.allclose(pred, exp, atol=1e-5):
            return False
    return True


def _dynamic_upscale2_oh(inp: np.ndarray) -> np.ndarray:
    occ = inp[0].max(axis=0) > 0.5
    if not occ.any():
        return np.zeros_like(inp)
    gh = int(np.where(occ.any(1))[0][-1]) + 1
    gw = int(np.where(occ.any(0))[0][-1]) + 1
    out = np.zeros_like(inp)
    for r in range(inp.shape[2]):
        for c in range(inp.shape[3]):
            if r < gh * 2 and c < gw * 2:
                sr, sc = r // 2, c // 2
                for k in range(10):
                    out[0, k, r, c] = inp[0, k, sr, sc]
    return out


def _dynamic_flip_oh(inp: np.ndarray, axis: str) -> np.ndarray:
    occ = inp[0].max(axis=0) > 0.5
    if not occ.any():
        return np.zeros_like(inp)
    gh = int(np.where(occ.any(1))[0][-1]) + 1
    gw = int(np.where(occ.any(0))[0][-1]) + 1
    out = np.zeros_like(inp)
    for k in range(10):
        reg = inp[0, k, :gh, :gw]
        if axis == "h":
            reg = reg[::-1, :]
        else:
            reg = reg[:, ::-1]
        out[0, k, :gh, :gw] = reg
    return out


def _verify_arcgen(fn, td, n: int = 100) -> bool:
    for ex in td.get("arc-gen", [])[:n]:
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        if not np.allclose(fn(inp), exp, atol=1e-5):
            return False
    return True


def s_bounded_upscale2(td):
    fn = _dynamic_upscale2_oh
    if not _verify_numpy(fn, td):
        return None
    if not _verify_arcgen(fn, td):
        return None
    return compile_dynamic_upscale2()


def s_bounded_flip_h(td):
    if not _verify_numpy(lambda x: _dynamic_flip_oh(x, "h"), td):
        return None
    return compile_dynamic_flip("h")


def s_bounded_flip_v(td):
    if not _verify_numpy(lambda x: _dynamic_flip_oh(x, "v"), td):
        return None
    return compile_dynamic_flip("v")


BOUNDED_SOLVERS = [
    ("bounded_upscale2", s_bounded_upscale2),
    ("bounded_flip_h", s_bounded_flip_h),
    ("bounded_flip_v", s_bounded_flip_v),
]

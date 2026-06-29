"""ARC-GEN-fitted object-program solvers (Phase 18) — rule transforms + gather compile."""

from __future__ import annotations

import numpy as np

from arc_genome.data.encoding import fixed_shapes
from arc_genome.genome.ops.arcgen_gather import (
    _arcgen_examples,
    s_position_gather_arcgen,
    s_spatial_gather_arcgen,
)


def _as_pairs(td) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array(ex["input"]), np.array(ex["output"])) for ex in _arcgen_examples(td)]


def _rule_then_gather_arcgen(td, rule_fn):
    """Apply a pixel rule on train+test+ARC-GEN; compile gather if consistent."""
    pairs = _as_pairs(td)
    if not pairs:
        return None
    if not all(np.array_equal(rule_fn(inp), out) for inp, out in pairs):
        return None
    return s_spatial_gather_arcgen(td) or s_position_gather_arcgen(td)


def _gravity_fn(direction: str):
    def fn(g: np.ndarray) -> np.ndarray:
        h, w = g.shape
        o = np.zeros_like(g)
        if direction == "down":
            for c in range(w):
                nz = g[:, c][g[:, c] != 0]
                if len(nz):
                    o[h - len(nz) :, c] = nz
        elif direction == "up":
            for c in range(w):
                nz = g[:, c][g[:, c] != 0]
                if len(nz):
                    o[: len(nz), c] = nz
        elif direction == "right":
            for r in range(h):
                nz = g[r, :][g[r, :] != 0]
                if len(nz):
                    o[r, w - len(nz) :] = nz
        elif direction == "left":
            for r in range(h):
                nz = g[r, :][g[r, :] != 0]
                if len(nz):
                    o[r, : len(nz)] = nz
        return o

    return fn


def _s_gravity_arcgen(td, direction: str):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    grav = _gravity_fn(direction)
    pairs = _as_pairs(td)
    if not all(np.array_equal(grav(inp), out) for inp, out in pairs):
        return None

    from arc_genome.onnx.gather import build_gather_model_with_const

    inp0, out0 = pairs[0]
    idx = np.full((ih, iw, 2), -1, dtype=np.int64)
    cst = np.full((ih, iw), -1, dtype=np.int64)
    if direction in ("down", "up"):
        for c in range(iw):
            src = [r for r in range(ih) if inp0[r, c] != 0]
            dst = [r for r in range(ih) if out0[r, c] != 0]
            if len(src) != len(dst):
                return None
            for dr, sr in zip(dst, src):
                idx[dr, c] = [sr, c]
            for r in range(ih):
                if out0[r, c] == 0:
                    cst[r, c] = 0
    else:
        for r in range(ih):
            src = [c for c in range(iw) if inp0[r, c] != 0]
            dst = [c for c in range(iw) if out0[r, c] != 0]
            if len(src) != len(dst):
                return None
            for dc, sc in zip(dst, src):
                idx[r, dc] = [r, sc]
            for c in range(iw):
                if out0[r, c] == 0:
                    cst[r, c] = 0

    for inp, out in pairs:
        pred = np.zeros_like(out)
        for r in range(ih):
            for c in range(iw):
                if idx[r, c, 0] >= 0:
                    pred[r, c] = inp[idx[r, c, 0], idx[r, c, 1]]
                elif cst[r, c] >= 0:
                    pred[r, c] = cst[r, c]
                else:
                    return None
        if not np.array_equal(pred, out):
            return None
    return build_gather_model_with_const(ih, iw, ih, iw, idx, cst)


def s_gravity_down_arcgen(td):
    return _s_gravity_arcgen(td, "down")


def s_gravity_up_arcgen(td):
    return _s_gravity_arcgen(td, "up")


def s_gravity_left_arcgen(td):
    return _s_gravity_arcgen(td, "left")


def s_gravity_right_arcgen(td):
    return _s_gravity_arcgen(td, "right")


def s_keep_color_arcgen(td):
    for color in range(1, 10):
        m = _rule_then_gather_arcgen(td, lambda inp, c=color: np.where(inp == c, c, 0))
        if m is not None:
            return m
    return None


def s_remove_color_arcgen(td):
    for color in range(1, 10):
        m = _rule_then_gather_arcgen(td, lambda inp, c=color: np.where(inp == c, 0, inp))
        if m is not None:
            return m
    return None


def s_largest_cc_arcgen(td):
    def largest(g):
        from collections import deque

        h, w = g.shape
        vis = np.zeros_like(g, dtype=bool)
        best: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0 or vis[r, c]:
                    continue
                col = int(g[r, c])
                q = deque([(r, c)])
                vis[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and g[nr, nc] == col:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                if len(cells) > len(best):
                    best = cells
        o = np.zeros_like(g)
        for r, c in best:
            o[r, c] = g[r, c]
        return o

    return _rule_then_gather_arcgen(td, largest)


def s_hollow_rect_arcgen(td):
    def hollow(g):
        o = np.zeros_like(g)
        for color in np.unique(g):
            if color == 0:
                continue
            m = g == color
            rs = np.where(m.any(1))[0]
            cs = np.where(m.any(0))[0]
            r0, r1, c0, c1 = rs[0], rs[-1], cs[0], cs[-1]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if r in (r0, r1) or c in (c0, c1):
                        o[r, c] = color
        return o

    return _rule_then_gather_arcgen(td, hollow)


def s_fill_rect_arcgen(td):
    def fill(g):
        o = np.zeros_like(g)
        for color in np.unique(g):
            if color == 0:
                continue
            m = g == color
            rs = np.where(m.any(1))[0]
            cs = np.where(m.any(0))[0]
            r0, r1, c0, c1 = rs[0], rs[-1], cs[0], cs[-1]
            o[r0 : r1 + 1, c0 : c1 + 1] = color
        return o

    return _rule_then_gather_arcgen(td, fill)


ARCgen_OBJECT_SOLVERS = [
    ("gravity_down_arcgen", s_gravity_down_arcgen),
    ("gravity_up_arcgen", s_gravity_up_arcgen),
    ("gravity_left_arcgen", s_gravity_left_arcgen),
    ("gravity_right_arcgen", s_gravity_right_arcgen),
    ("keep_color_arcgen", s_keep_color_arcgen),
    ("remove_color_arcgen", s_remove_color_arcgen),
    ("largest_cc_arcgen", s_largest_cc_arcgen),
    ("hollow_rect_arcgen", s_hollow_rect_arcgen),
    ("fill_rect_arcgen", s_fill_rect_arcgen),
]

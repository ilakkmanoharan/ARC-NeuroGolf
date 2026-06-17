"""Milestone 5 analytical solvers — variable input + object primitives."""

from __future__ import annotations

import numpy as np

from arc_genome.data.encoding import GH, GW, fixed_shapes, get_examples
from arc_genome.onnx.gather import build_gather_model, build_gather_model_with_const


def _fixed_output_dims(td) -> tuple[int, int] | None:
    shapes = {tuple(out.shape) for _, out in get_examples(td)}
    return shapes.pop() if len(shapes) == 1 else None


def _position_idx(td) -> tuple[np.ndarray | None, int, int]:
    dims = _fixed_output_dims(td)
    if dims is None:
        return None, 0, 0
    oh, ow = dims
    exs = get_examples(td)
    idx = np.zeros((oh, ow, 2), dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            found = False
            for sr in range(GH):
                for sc in range(GW):
                    if all(
                        sr < inp.shape[0]
                        and sc < inp.shape[1]
                        and int(out[r, c]) == int(inp[sr, sc])
                        for inp, out in exs
                    ):
                        idx[r, c] = [sr, sc]
                        found = True
                        break
                if found:
                    break
            if not found:
                return None, 0, 0
    return idx, oh, ow


def s_position_gather_var(td):
    """Position gather with variable input shapes (fixed output shape required)."""
    if fixed_shapes(td) is not None:
        return None
    idx, oh, ow = _position_idx(td)
    if idx is None:
        return None
    return build_gather_model(oh, ow, idx)


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


def _s_gravity_dir(td, direction: str):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    if (ih, iw) != (oh, ow):
        return None
    grav = _gravity_fn(direction)
    exs = get_examples(td)
    if not all(np.array_equal(grav(inp), out) for inp, out in exs):
        return None

    inp0, out0 = exs[0]
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

    for inp, out in exs:
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


def s_gravity_down_var(td):
    return _s_gravity_dir(td, "down")


def s_gravity_up_var(td):
    return _s_gravity_dir(td, "up")


def s_gravity_left_var(td):
    return _s_gravity_dir(td, "left")


def s_gravity_right_var(td):
    return _s_gravity_dir(td, "right")


def _rule_then_gather(td, rule_fn):
    exs = get_examples(td)
    if not all(np.array_equal(rule_fn(inp), out) for inp, out in exs):
        return None
    return s_position_gather_var(td) or _position_gather_fixed(td)


def _position_gather_fixed(td):
    sp = fixed_shapes(td)
    if sp is None:
        return None
    from arc_genome.genome.ops.analytical import s_position_gather
    return s_position_gather(td)


def s_keep_color(td):
    for color in range(1, 10):
        m = _rule_then_gather(td, lambda inp, c=color: np.where(inp == c, c, 0))
        if m is not None:
            return m
    return None


def s_remove_color(td):
    for color in range(1, 10):
        m = _rule_then_gather(td, lambda inp, c=color: np.where(inp == c, 0, inp))
        if m is not None:
            return m
    return None


def s_replace_background(td):
    for color in range(1, 10):
        m = _rule_then_gather(td, lambda inp, c=color: np.where(inp == 0, c, inp))
        if m is not None:
            return m
    return None


def _bbox(g: np.ndarray):
    m = g != 0
    if not m.any():
        return None
    rs = np.where(m.any(1))[0]
    cs = np.where(m.any(0))[0]
    return int(rs[0]), int(rs[-1]) + 1, int(cs[0]), int(cs[-1]) + 1


def s_extract_bbox(td):
    exs = get_examples(td)
    if _fixed_output_dims(td) is None:
        return None
    for inp, out in exs:
        b = _bbox(inp)
        if b is None:
            return None
        r0, r1, c0, c1 = b
        if not np.array_equal(inp[r0:r1, c0:c1], out):
            return None
    return s_position_gather_var(td)


def s_hollow_rect(td):
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

    return _rule_then_gather(td, hollow)


def s_fill_rect(td):
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

    return _rule_then_gather(td, fill)


def s_largest_component(td):
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

    return _rule_then_gather(td, largest)


MILESTONE5_SOLVERS = [
    ("position_gather_var", s_position_gather_var),
    ("gravity_down", s_gravity_down_var),
    ("gravity_up", s_gravity_up_var),
    ("gravity_left", s_gravity_left_var),
    ("gravity_right", s_gravity_right_var),
    ("keep_color", s_keep_color),
    ("remove_color", s_remove_color),
    ("replace_bg", s_replace_background),
    ("extract_bbox", s_extract_bbox),
    ("hollow_rect", s_hollow_rect),
    ("fill_rect", s_fill_rect),
    ("largest_cc", s_largest_component),
]

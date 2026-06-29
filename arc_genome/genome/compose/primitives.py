"""Numpy grid primitives for composition search (phase 4)."""

from __future__ import annotations

from typing import Callable

import numpy as np

GridFn = Callable[[np.ndarray], np.ndarray]


def _identity(g: np.ndarray) -> np.ndarray:
    return g.copy()


def _flip_h(g: np.ndarray) -> np.ndarray:
    return np.flipud(g)


def _flip_v(g: np.ndarray) -> np.ndarray:
    return np.fliplr(g)


def _rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 1)


def _rot180(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 2)


def _transpose(g: np.ndarray) -> np.ndarray:
    return g.T


def _tile2x2(g: np.ndarray) -> np.ndarray:
    return np.tile(g, (2, 2))


def _upscale2(g: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(g, 2, 0), 2, 1)


def _crop_center(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    ch, cw = h // 2, w // 2
    return g[ch : h - ch, cw : w - cw] if ch and cw else g


def _gravity_down(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    o = np.zeros_like(g)
    for c in range(w):
        nz = g[:, c][g[:, c] != 0]
        if len(nz):
            o[h - len(nz) :, c] = nz
    return o


def _gravity_up(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    o = np.zeros_like(g)
    for c in range(w):
        nz = g[:, c][g[:, c] != 0]
        if len(nz):
            o[: len(nz), c] = nz
    return o


def _gravity_left(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    o = np.zeros_like(g)
    for r in range(h):
        nz = g[r, :][g[r, :] != 0]
        if len(nz):
            o[r, : len(nz)] = nz
    return o


def _gravity_right(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    o = np.zeros_like(g)
    for r in range(h):
        nz = g[r, :][g[r, :] != 0]
        if len(nz):
            o[r, w - len(nz) :] = nz
    return o


def _largest_cc(g: np.ndarray) -> np.ndarray:
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


def _scale_down2(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    if h % 2 or w % 2:
        raise ValueError("not divisible by 2")
    return g[::2, ::2]


def _scale_down3(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    if h % 3 or w % 3:
        raise ValueError("not divisible by 3")
    return g[::3, ::3]


def apply_color_map(g: np.ndarray, cm: dict[int, int]) -> np.ndarray:
    out = g.copy()
    for c in np.unique(g):
        ci = int(c)
        if ci in cm:
            out[g == c] = cm[ci]
    return out


def fit_color_map(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, int] | None:
    cm: dict[int, int] = {}
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        for iv, ov in zip(inp.flat, out.flat):
            iv, ov = int(iv), int(ov)
            if iv in cm and cm[iv] != ov:
                return None
            cm[iv] = ov
    return cm if cm else None


def apply_chain_fns(grid: np.ndarray, fns: tuple) -> np.ndarray:
    out = grid
    for fn in fns:
        out = fn(out)
    return out


PRIMITIVE_FUNCS: dict[str, GridFn] = {
    "identity": _identity,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "rot90": _rot90,
    "rot180": _rot180,
    "transpose": _transpose,
    "tile2x2": _tile2x2,
    "upscale2": _upscale2,
    "crop_center": _crop_center,
    "gravity_down": _gravity_down,
    "gravity_up": _gravity_up,
    "gravity_left": _gravity_left,
    "gravity_right": _gravity_right,
    "largest_cc": _largest_cc,
    "scale_down2": _scale_down2,
    "scale_down3": _scale_down3,
}

PRIMITIVE_NAMES = list(PRIMITIVE_FUNCS.keys())


def apply_chain(grid: np.ndarray, chain: tuple[str, ...]) -> np.ndarray:
    ops = PRIMITIVE_FUNCS
    out = grid
    for name in chain:
        out = ops[name](out)
    return out

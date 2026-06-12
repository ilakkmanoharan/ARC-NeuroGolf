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


def apply_chain(grid: np.ndarray, chain: tuple[str, ...]) -> np.ndarray:
    ops = PRIMITIVE_FUNCS
    out = grid
    for name in chain:
        out = ops[name](out)
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
}

PRIMITIVE_NAMES = list(PRIMITIVE_FUNCS.keys())

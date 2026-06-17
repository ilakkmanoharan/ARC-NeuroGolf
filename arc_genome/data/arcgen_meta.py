"""ARC-GEN signature detection for belief-guided solver routing (Phase 13)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_genome.genome.compose.primitives import PRIMITIVE_FUNCS

# Solver names that arc-gen signatures map to (harness family priors).
SIGNATURE_SOLVERS: dict[str, list[str]] = {
    "upscale2": ["bounded_upscale2", "upscale"],
    "flip_h": ["bounded_flip_h", "flip"],
    "flip_v": ["bounded_flip_v", "flip"],
    "color_map": ["color_map"],
    "transpose": ["transpose"],
    "identity": ["identity"],
    "scale_down": ["scale_down"],
    "translate": ["translate"],
    "gravity_down": ["gravity_down", "gravity"],
}


def _examples(td, n: int = 50) -> list[dict]:
    return td.get("train", []) + td.get("test", []) + td.get("arc-gen", [])[:n]


def _match_primitive(examples: list[dict], name: str) -> float:
    fn = PRIMITIVE_FUNCS[name]
    if not examples:
        return 0.0
    hits = 0
    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        try:
            if np.array_equal(fn(inp), out):
                hits += 1
        except Exception:
            pass
    return hits / len(examples)


def _match_upscale2(examples: list[dict]) -> float:
    if not examples:
        return 0.0
    hits = 0
    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        ih, iw = inp.shape
        if out.shape != (ih * 2, iw * 2):
            continue
        if np.array_equal(out, np.repeat(np.repeat(inp, 2, 0), 2, 1)):
            hits += 1
    return hits / len(examples)


def _match_flip(examples: list[dict], axis: str) -> float:
    if not examples:
        return 0.0
    fn = np.fliplr if axis == "h" else np.flipud
    hits = sum(1 for ex in examples if np.array_equal(fn(np.array(ex["input"])), np.array(ex["output"])))
    return hits / len(examples)


def _match_color_map(examples: list[dict]) -> float:
    if not examples:
        return 0.0
    cm: dict[int, int] = {}
    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        if inp.shape != out.shape:
            return 0.0
        for iv, ov in zip(inp.flat, out.flat):
            iv, ov = int(iv), int(ov)
            if iv in cm and cm[iv] != ov:
                return 0.0
            cm[iv] = ov
    return 1.0


def detect_signatures(td, n: int = 50) -> list[tuple[str, float]]:
    """Return (signature_name, confidence) sorted descending."""
    exs = _examples(td, n)
    scores: Counter[str] = Counter()

    u2 = _match_upscale2(exs)
    if u2 > 0.5:
        scores["upscale2"] = u2

    fh = _match_flip(exs, "h")
    if fh > 0.5:
        scores["flip_h"] = fh
    fv = _match_flip(exs, "v")
    if fv > 0.5:
        scores["flip_v"] = fv

    cm = _match_color_map(exs)
    if cm > 0.9:
        scores["color_map"] = cm

    for name in ("transpose", "identity"):
        m = _match_primitive(exs, name)
        if m > 0.9:
            scores[name] = m

    return sorted(scores.items(), key=lambda x: -x[1])


def solver_prior(td, n: int = 50) -> list[tuple[str, float]]:
    """Map arc-gen signatures to ordered solver-name priors."""
    priors: Counter[str] = Counter()
    for sig, conf in detect_signatures(td, n):
        for solver in SIGNATURE_SOLVERS.get(sig, []):
            priors[solver] = max(priors[solver], conf)
    return sorted(priors.items(), key=lambda x: -x[1])

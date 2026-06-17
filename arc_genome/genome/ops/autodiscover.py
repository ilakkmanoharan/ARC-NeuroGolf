"""ARC-GEN autodiscovery: numpy scan → bounded solver selection (Phase 14)."""

from __future__ import annotations

import numpy as np

from arc_genome.data.encoding import to_onehot

BOUNDED_RULES: list[tuple[str, object]] = []


def _register(name: str):
    def deco(fn):
        BOUNDED_RULES.append((name, fn))
        return fn
    return deco


def _occ(inp):
    return inp[0].max(axis=0) > 0.5


def _bbox(inp):
    occ = _occ(inp)
    if not occ.any():
        return 0, 0
    gh = int(np.where(occ.any(1))[0][-1]) + 1
    gw = int(np.where(occ.any(0))[0][-1]) + 1
    return gh, gw


def _apply_reg(inp, reg_fn, shape_ok=True):
    gh, gw = _bbox(inp)
    if gh == 0:
        return np.zeros_like(inp)
    out = np.zeros_like(inp)
    for k in range(10):
        reg = inp[0, k, :gh, :gw]
        treg = reg_fn(reg)
        if shape_ok and treg.shape != reg.shape:
            return None
        oh, ow = min(treg.shape[0], 30), min(treg.shape[1], 30)
        out[0, k, :oh, :ow] = treg[:oh, :ow]
    return out


@_register("rule_flip_h")
def rule_flip_h(inp):
    return _apply_reg(inp, lambda r: r[:, ::-1])


@_register("rule_flip_v")
def rule_flip_v(inp):
    return _apply_reg(inp, lambda r: r[::-1, :])


@_register("rule_upscale2")
def rule_upscale2(inp):
    gh, gw = _bbox(inp)
    if gh == 0:
        return np.zeros_like(inp)
    out = np.zeros_like(inp)
    for r in range(30):
        for c in range(30):
            if r < gh * 2 and c < gw * 2:
                sr, sc = r // 2, c // 2
                for k in range(10):
                    out[0, k, r, c] = inp[0, k, sr, sc]
    return out


@_register("rule_rot90")
def rule_rot90(inp):
    return _apply_reg(inp, lambda r: np.rot90(r, 1), shape_ok=False)


@_register("rule_rot180")
def rule_rot180(inp):
    return _apply_reg(inp, lambda r: np.rot90(r, 2))


@_register("rule_transpose")
def rule_transpose(inp):
    return _apply_reg(inp, lambda r: r.T, shape_ok=False)


@_register("rule_scale_down2")
def rule_scale_down2(inp):
    gh, gw = _bbox(inp)
    if gh < 2 or gw < 2:
        return None
    out = np.zeros_like(inp)
    oh, ow = gh // 2, gw // 2
    if oh == 0 or ow == 0:
        return None
    for k in range(10):
        out[0, k, :oh, :ow] = inp[0, k, : oh * 2, : ow * 2][::2, ::2]
    return out


RULE_TO_SOLVER = {
    "rule_flip_h": "bounded_flip_h",
    "rule_flip_v": "bounded_flip_v",
    "rule_upscale2": "bounded_upscale2",
    "rule_rot90": "bounded_rot90",
    "rule_rot180": "bounded_rot180",
    "rule_transpose": "bounded_transpose",
    "rule_scale_down2": "bounded_scale_down2",
}


def discover_rules(td, n_arcgen: int = 100) -> list[tuple[str, float]]:
    """Return (solver_name, confidence) for rules passing train+test+arc-gen."""
    exs = td["train"] + td["test"] + td.get("arc-gen", [])[:n_arcgen]
    hits: list[tuple[str, float]] = []
    for name, fn in BOUNDED_RULES:
        ok = True
        for ex in exs:
            pred = fn(to_onehot(ex["input"]))
            if pred is None or not np.allclose(pred, to_onehot(ex["output"]), atol=1e-5):
                ok = False
                break
        if ok:
            solver = RULE_TO_SOLVER.get(name, name)
            hits.append((solver, 1.0))
    return hits


def prescan_new_tasks(solved: set[int], tasks: dict) -> list[int]:
    """Tasks not in solved set that autodiscover rules pass."""
    new = []
    for tn, meta in tasks.items():
        if tn in solved:
            continue
        if discover_rules(meta["data"]):
            new.append(tn)
    return sorted(new)

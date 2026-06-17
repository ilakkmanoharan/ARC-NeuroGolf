"""BLF-style belief ranking over transform hypotheses (Phase 11/13)."""

from __future__ import annotations

import numpy as np

from arc_genome.data.encoding import to_onehot


def _train_match_score(fn, td) -> float:
    n = len(td["train"]) + len(td["test"])
    if n == 0:
        return 0.0
    hits = 0
    for ex in td["train"] + td["test"]:
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        try:
            if np.allclose(fn(inp), exp, atol=1e-5):
                hits += 1
        except Exception:
            pass
    return hits / n


def rank_bounded_hypotheses(td) -> list[tuple[str, float]]:
    """Return (solver_name, belief) sorted descending."""
    from arc_genome.genome.ops.bounded import _dynamic_flip_oh, _dynamic_upscale2_oh

    hyps = [
        ("bounded_upscale2", lambda x: _dynamic_upscale2_oh(x)),
        ("bounded_flip_h", lambda x: _dynamic_flip_oh(x, "h")),
        ("bounded_flip_v", lambda x: _dynamic_flip_oh(x, "v")),
    ]
    scored = [(name, _train_match_score(fn, td)) for name, fn in hyps]
    return sorted(scored, key=lambda x: -x[1])


def rank_solver_order(td, solver_names: list[str]) -> list[str]:
    """Reorder solver list using ARC-GEN priors + train-match bounded beliefs."""
    from arc_genome.config import get_config
    from arc_genome.data.arcgen_meta import solver_prior

    if not get_config().arcgen_routing:
        return solver_names

    scores: dict[str, float] = {n: 0.0 for n in solver_names}
    for name, conf in solver_prior(td):
        if name in scores:
            scores[name] = max(scores[name], conf)
    for name, conf in rank_bounded_hypotheses(td):
        if name in scores and conf > 0:
            scores[name] = max(scores[name], conf + 0.1)

    ranked = sorted(solver_names, key=lambda n: (-scores.get(n, 0.0), solver_names.index(n)))
    return ranked

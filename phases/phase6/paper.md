# Phase 6: Generalization Hardening + Cost Audit

## Abstract

Phase 6 is the **leaderboard optimization** layer: ARC-GEN synthetic validation prevents overfit conv solutions from scoring 0 on private tests, and a full cost audit replaces expensive wins with cheaper equivalents.

## Theory

### The overfitting problem

Conv least-squares on 3–4 train pairs can memorize without learning φ. Kaggle validates against:
- Original ARC train/test
- ARC-GEN-100K synthetic variants
- Private held-out suite

A network that passes local train/test but fails ARC-GEN likely earns **zero** competition points.

### ARC-GEN validation gate

```text
accept(model) ⟺ verify(train) ∧ verify(arc_gen_samples)
```

If `data/arc_gen.json` is unavailable, gate gracefully passes (no false rejections).

### Cost audit

For every task with an existing solution:

```text
new_candidates = full_solver_chain(task)
if cost(new) < cost(old):
    replace ONNX
```

Phase 6 re-runs the complete pipeline with all prior phases enabled, keeping improvements only.

### Submission hygiene

- Zip includes only validated ONNX files
- Stale failed-conv artifacts excluded
- Per-phase `results.json` logs solver distribution

### What was special

Phase 6 optimizes the **expected Kaggle score** E[points], not just local solve count. A conv that locally validates but fails generalization is worthless.

### Target

Combined with Phases 1–5: **7,000+ Kaggle score** (400 tasks × ~17–19 avg).

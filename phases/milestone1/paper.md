# Milestone 1: Fix Measurement — Stop Flying Blind

**Target:** 388 → 500 Kaggle score  
**Theme:** Calibrated scoring + ARC-GEN validation + curated submission

## What We Discovered

The official Kaggle scorer (from `rogermt/neurogolf-solver`) uses:

```text
cost  = memory_bytes + params        (NOT MACs in score formula)
score = max(1, 25 − ln(cost))
memory = ONNX Runtime profiler peak tensor footprints
```

Our local `cost.py` was wrong in two ways:
1. Added MACs to cost (not in official score)
2. Estimated memory instead of profiling

This explains the 5.7× gap: local predicted ~2,230, Kaggle gave 388.

## What Milestone 1 Implements

| Component | Purpose |
|---|---|
| `arc_genome/onnx/kaggle_score.py` | Official ORT-profiler scoring |
| `data/arc_gen_raw/` | ARC-GEN-100K (6 samples × 400 tasks) |
| `scripts/audit_submission.py` | Tier tags: pass_all / train_only / fail |
| `scripts/run_milestone1.py` | Audit → curate → submit |

## Validation Tiers

```text
pass_all     train + test + arc-gen (30 samples) ✓
train_only   train + test ✓, arc-gen ✗  → likely scores 0 on Kaggle
fail         does not pass train/test
```

## Curated Submission Strategy

Include only `pass_all` tasks with official `kaggle_score ≥ threshold`.
Drop train-only overfits that inflate zip size but earn 0 points.

## Success Criteria

- Official scorer matches Kaggle within ~10% on probe models
- Know true scoring task count vs solve count
- Kaggle score ≥ 500

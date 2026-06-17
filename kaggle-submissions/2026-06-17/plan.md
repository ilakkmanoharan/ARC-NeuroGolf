# Submission Plan — 2026-06-17 (M7b)

**Baseline:** M6 — 64 tasks, 834.44 Kaggle  
**Target:** 64 tasks, **835+** Kaggle (score-per-task optimization)

---

## Objective

No new coverage expected (gravity fails ARC-GEN). Today's submission optimizes **circuit cost** on existing pass_all tasks.

## Primary change: Slim bounded flip (Phase 12)

| Metric | M6 flip | M7b slim flip |
|---|---|---|
| ONNX nodes | 11,215 | 11,168 |
| Kaggle cost (tasks 150/155) | 252,810 | 179,836 |
| Kaggle score per flip task | 12.56 | **12.90** |
| pass_all | ✓ | ✓ |

**Expected total gain:** +0.68 points (2 tasks × +0.34)

## Secondary

- Inherits Phase 11 pipeline (official score, 100 ARC-GEN, bounded solvers first)
- No third-pass changes
- No ARC-GEN gate relaxation

## Out of scope (documented, not submitted)

- Bounded gravity — task 32 train-only, fails ARC-GEN
- Duplicate 64-task zip without changes

---

## Run

```bash
python scripts/run_submission_2026-06-17.py
```

Outputs to `kaggle-submissions/2026-06-17/`.

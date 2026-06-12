# Milestone 1 Results

## Kaggle submission

- **Submitted:** `submission.zip` (23 tasks, ARC-GEN verified)
- **Estimated score:** 388.2 (matches previous 388.20 exactly)

## Breakthrough finding

Audit of Phase 6 (199 ONNX files):

| Tier | Count | Meaning |
|---|---|---|
| **pass_all** | **23** | Pass train + test + ARC-GEN → **earn Kaggle points** |
| train_only | 176 | Pass train/test, **fail ARC-GEN** → score 0 on Kaggle |
| fail | 0 | Don't pass train/test |

```text
pass_all official score sum = 388.20  ← exactly matches Kaggle leaderboard
```

**176 of 199 "solved" tasks were overfit convolutions** that never generalized.

## Official scorer calibrated

| Task | Local score | Official Kaggle score |
|---|---|---|
| task016 (color_map) | 13.0 | **20.4** |

Official formula: `score = max(1, 25 − ln(memory + params))` using ORT profiler.

## Files

- `paper.md` — theory
- `audit.json` — per-task tiers and official scores
- `curated.json` — 23 included tasks
- `run.log` — full run output

## Milestone 1b (re-solve with ARC-GEN gate)

- **Solved:** 52/400 tasks (43 min)
- **pass_all:** 52 (all ARC-GEN verified)
- **Estimated Kaggle score:** **843.4** (up from 388.2)
- **Zip:** `submission_v2.zip` — **not yet submitted** (Kaggle API 400; likely daily limit after 14 submissions today)

```bash
~/Library/Python/3.9/bin/kaggle competitions submit -c neurogolf-2026 \
  -f phases/milestone1/submission_v2.zip \
  -m "ARC-Genome M1b: 52 ARC-GEN verified tasks"
```

## Next (Milestone 2)

1. Submit M1b when rate limit resets
2. Debug extended analytical solvers (0 wins in phases 3–4)
3. Target: grow pass_all from 52 → 80+

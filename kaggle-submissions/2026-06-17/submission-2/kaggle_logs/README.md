# Kaggle Logs — submission-2

**Kaggle score:** 848.07 | **Audit est.:** 998.55 | **Ratio:** 84.93%

---

## Leaderboard (from Kaggle UI)

| Time | Message | Tasks | Score |
|---|---|---|---|
| 39m ago | ARC-Genome M8: 65 verified, est 999, arcgen routing | 65 | **848.07** |
| 3h ago | ARC-Genome M7b: 64 verified, est 986, slim bounded flip | 64 | 835.12 |
| 3d ago | ARC-Genome M6: 64 verified, est 985, bounded CWM flip | 64 | 834.44 |
| 3d ago | ARC-Genome M5: 62 verified, est 960, var-shape solvers | 62 | 809.32 |

---

## Per-task scoring log

Generated via `scripts/fetch_kaggle_logs.py` using official `kaggle_score_model` scorer.

| Metric | Value |
|---|---|
| Tasks scored | 65 |
| pass_all | 65 |
| Est. total | 998.55 |
| Actual total | **848.07** |
| Audit ratio | **0.8493** |

Full per-task breakdown: `kaggle_logs/kaggle_logs.json`  
Official task JSONs (Kaggle): `kaggle_logs/official_tasks/` (downloaded per task)

---

## Score delta validation

```text
submission-1: 835.12 (64 tasks)
submission-2: 848.07 (65 tasks)
Delta:        +12.95

Task 307 (bounded_upscale2) audit score: 12.94
→ Delta is exactly one new task. Theory validated.
```

---

## Lowest-scoring tasks (score improvement targets)

| Task | Solver | Score | Memory |
|---|---|---|---|
| 150 | bounded_flip_v | 12.90 | 179,220 |
| 155 | bounded_flip_h | 12.90 | 179,220 |
| 307 | bounded_upscale2 | 12.94 | 172,028 |
| 53, 87, 116… | position_gather/compose | 13.39 | 108,000 |

Bounded CWM tasks are the **score floor** — gh/gw unroll dominates memory.

---

## Autodiscover cross-check (post-score)

Rules passing 100 ARC-GEN on solved tasks:

| Rule | Tasks | Current solver | Notes |
|---|---|---|---|
| bounded_flip_h/v | 150, 155 | bounded_flip | Already optimal family |
| bounded_upscale2 | 307 | bounded_upscale2 | New in M8 |
| bounded_rot180 | 87, 140 | compose (13.39) | Bounded compile would be worse (~12.9) |
| bounded_transpose | 179, 241 | transpose (**25.0**) | Analytical wins |
| bounded_rot90 | 380 | compose (13.39) | s_rotate same score |

**0 autodiscover rules on 335 unsolved tasks** — coverage ceiling is real.

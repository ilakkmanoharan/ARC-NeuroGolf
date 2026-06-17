# Submission Analysis — submission-1 (M7b)

**Date:** 2026-06-17  
**Kaggle score:** **835.12** ✅  
**Tasks:** 64 pass_all  
**Message:** `ARC-Genome M7b: 64 verified, est 986, slim bounded flip`

---

## Score progression

| Submission | Tasks | Kaggle | Δ |
|---|---|---|---|
| M6 (2026-06-14) | 64 | 834.44 | — |
| **submission-1 (M7b)** | **64** | **835.12** | **+0.68** |

Prediction was exact: +0.34 per flip task × 2 = +0.68.

---

## What changed vs M6

**Phase 12 slim bounded flip** — single `Gather` on `[1, 10, 900]` instead of 10 per-channel Gathers.

| Task | Solver | M6 score | M7b score | M6 cost | M7b cost |
|---|---|---|---|---|---|
| 150 | bounded_flip_v | 12.56 | **12.90** | 252,810 | **179,836** |
| 155 | bounded_flip_h | 12.56 | **12.90** | 252,810 | **179,836** |

All other 62 tasks unchanged (same ONNX files selected).

---

## Audit vs Kaggle

| Metric | Value |
|---|---|
| Audit est. | 985.6 |
| Kaggle actual | **835.12** |
| Ratio | 84.7% |

Same ~15% overestimate as M3/M6 — local audit is optimistic but rank-ordering is correct.

---

## Solver mix (64 tasks)

| Solver | Count |
|---|---|
| conv_diff | 19 |
| conv_fixed | 14 |
| position_gather | 7 |
| conv_var | 7 |
| concat | 5 |
| color_map | 4 |
| compose | 3 |
| transpose | 2 |
| bounded_flip_v | 1 |
| bounded_flip_h | 1 |
| upscale | 1 |

**Avg Kaggle score:** 835.12 / 64 = **13.05** pts/task

---

## Run stats

- Solve time: 5,285s (~88 min)
- Third pass: 336 unsolved → 0 new wins
- Zip: 530 KB, 64 ONNX files

---

## Artifacts in this folder

| File | Description |
|---|---|
| `run.log` | Full solve + audit + submit log |
| `audit.json` | Per-task official scores |
| `results.json` | Summary JSON |
| `kaggle_status.txt` | Kaggle leaderboard snapshot |
| `submission/` | 64 verified ONNX files |
| `submission_v2.zip` | Kaggle upload zip |
| `theory.md` | Phase 12 theory |
| `analysis.md` | This file |
| `learnings.md` | Distilled takeaways for submission-2 |

---

## Takeaway

**Cost optimization without new coverage works.** Same 64 tasks, +0.68 Kaggle by slimming bounded flip graphs. Next lever for meaningful jumps: new task coverage (M8 ARC-GEN routing).

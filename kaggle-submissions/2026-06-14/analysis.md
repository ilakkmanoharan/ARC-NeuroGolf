# Submission Analysis — 2026-06-14 (M6)

**Kaggle score:** 834.44  
**Tasks:** 64 pass_all  
**Message:** `ARC-Genome M6: 64 verified, est 985, bounded CWM flip`

---

## Score progression

| Milestone | Tasks | Kaggle | Δ |
|---|---|---|---|
| M3–M5 | 62 | 809.32 | — |
| **M6** | **64** | **834.44** | **+25.12** |

First ceiling break since M3. +2 tasks from bounded dynamic flip (150, 155).

---

## Solver mix (64 tasks)

| Solver | Count | Avg Kaggle score |
|---|---|---|
| conv_diff | 19 | ~15.5 |
| conv_fixed | 14 | ~15.0 |
| position_gather | 7 | ~13.4 |
| conv_var | 7 | ~15.0 |
| concat | 5 | ~18 |
| color_map | 4 | ~20 |
| compose | 3 | ~17 |
| transpose | 2 | ~20 |
| **bounded_flip_v** | 1 | **12.56** |
| **bounded_flip_h** | 1 | **12.56** |
| upscale | 1 | ~18 |

---

## Audit vs Kaggle

| Metric | Value |
|---|---|
| Audit est. | 984.9 |
| Kaggle actual | 834.44 |
| Ratio | 84.7% |

Consistent with M3 pattern (est ~19% high).

---

## Key wins

- **Task 150** (`bounded_flip_v`): variable-shape fliplr — first dynamic bbox CWM win
- **Task 155** (`bounded_flip_h`): variable-shape flipud

## Bottleneck discovered

Bounded flip graphs are **expensive**:
- 11,215 ONNX nodes per model
- Kaggle cost 252,810 (memory 251,220)
- Score only **12.56** per task — drags average down

Tasks 150/155 are the lowest-scoring pass_all tasks in the submission.

---

## Run stats

- Solve time: 5,966s (~99 min)
- Third pass: 336 unsolved, 0 new wins
- Zip: 543 KB, 64 files

---

## Artifacts

- `run.log` — full solve output
- `audit.json` — per-task official scores
- `results.json` — solver counts
- `kaggle_status.txt` — leaderboard snapshot

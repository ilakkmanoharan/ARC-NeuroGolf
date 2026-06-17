# Milestone 3 Results

## Kaggle submission

- **Submitted:** 2026-06-14
- **Tasks:** 62 (all pass_all)
- **Est. score:** 959.8
- **Kaggle score:** **809.32** (was 761.25)
- **Improvement:** +48 points, +12 tasks

## Changes from M2

1. Official Kaggle score for candidate selection
2. ARC-GEN-aware bbox normalization for conv fitting
3. 100 ARC-GEN samples in conv least-squares fitting
4. New `position_gather` analytical solver

## Solver mix

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
| upscale | 1 |

## Files

- `private/milestone3-plan.md` — plan and rationale
- `audit.json`, `results.json`, `run.log`

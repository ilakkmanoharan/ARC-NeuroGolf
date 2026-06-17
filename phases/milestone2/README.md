# Milestone 2 Results

## Kaggle submission

- **Submitted:** 2026-06-14 — `submission_v2.zip` (50 tasks)
- **Message:** `ARC-Genome M2: 50 verified, est 777, 100 arcgen`
- **Est. score:** 777.1
- **Kaggle score:** **761.25** (+197 vs M1b 564.20)

## What changed

1. **Fixed gather ONNX** — `GatherElements` → `Gather` (opset 10 compatible)
2. **100 ARC-GEN samples** at solve-time gate (was 30)
3. **Analytical priority** — `prefer_structural` active in infer

## Solver mix

| Solver | Count |
|---|---|
| conv_fixed | 13 |
| conv_diff | 8 |
| spatial_gather | 7 |
| conv_var | 7 |
| concat | 5 |
| color_map | 4 |
| compose | 3 |
| transpose | 2 |
| upscale | 1 |

**24 analytical / 26 conv** — first submission with meaningful analytical coverage.

## Files

- `private/milestone2-plan.md` — theory and fixes
- `audit.json`, `results.json`, `run.log`

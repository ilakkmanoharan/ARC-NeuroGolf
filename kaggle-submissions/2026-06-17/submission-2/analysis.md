# Submission Analysis — submission-2 (M8)

**Date:** 2026-06-17  
**Kaggle:** 848.07 ✅  
**Tasks:** 65 pass_all  
**Message:** `ARC-Genome M8: 65 verified, est 999, arcgen routing`

---

## Score progression

| Submission | Tasks | Audit est. | Kaggle actual | Δ tasks |
|---|---|---|---|---|
| submission-1 (M7b) | 64 | 985.6 | 835.12 | — |
| **submission-2 (M8)** | **65** | **998.5** | **848.07** | **+1** |

Predicted Kaggle delta: +12.94 (task 307). **Actual: +12.95 — exact match.**

---

## What changed vs submission-1

**Phase 13 ARC-GEN routing** — belief-ordered solver enumeration from arc-gen signatures.

**New bounded primitive: `bounded_upscale2`** — dynamic 2× upscale within runtime content bbox.

| Task | Hex | Solver | Score | Cost |
|---|---|---|---|---|
| **307** | c59eb873 | **bounded_upscale2** | **12.94** | 172,705 |

All other 64 tasks retained (same solver families; minor cost drift from re-solve).

---

## Pre-scan validation

Numpy pre-scan before full solve identified exactly 1 new candidate:

```text
[307] → bounded_upscale2 pass_all
```

Full run confirmed: `Task 307: bounded_upscale2 cost=172,705 score=12.9`

---

## Solver mix (65 tasks)

| Solver | Count | Δ vs s1 |
|---|---|---|
| conv_diff | 19 | — |
| conv_fixed | 14 | — |
| position_gather | 7 | — |
| conv_var | 7 | — |
| concat | 5 | — |
| color_map | 4 | — |
| compose | 3 | — |
| transpose | 2 | — |
| bounded_flip_v | 1 | — |
| bounded_flip_h | 1 | — |
| upscale | 1 | — |
| **bounded_upscale2** | **1** | **+1** |

---

## Run stats

- Solve time: 5,219s (~87 min)
- Third pass: 335 unsolved → 0 new wins
- Zip: 587 KB, 65 ONNX files
- Pre-scan → full solve → audit → **submitted** (pass_all 65 > 64)

---

## Audit vs expected Kaggle

| Metric | Value |
|---|---|
| Audit est. | 998.5 |
| Kaggle actual | **848.07** |
| Ratio | **84.9%** |

---

## What didn't work

| Target | Result |
|---|---|
| ≥68 tasks | 65 (missed by 3) |
| mask_preserve / position_gather arcgen fixes | 0 new (rule doesn't generalize) |
| Additional bounded flip tasks | 0 new |
| Third-pass conv | 0 new (again) |

ARC-GEN routing successfully directed search; the win came from a **new bounded CWM primitive** (variable-shape upscale), not reordering alone.

---

## Artifacts

| File | Description |
|---|---|
| `run.log` | Full solve + audit log |
| `audit.json` | Per-task official scores |
| `results.json` | Summary JSON |
| `submission/` | 65 verified ONNX files |
| `submission_v2.zip` | Kaggle upload zip |
| `theory.md` | Phase 13 theory |
| `kaggle_notebook.md` | Repro workflow |
| `analysis.md` | This file |

---

## Takeaway

**Coverage +1 task via arc-gen-guided discovery of variable-shape upscale.** Routing infrastructure is in place; next wins need more bounded primitives (rotate, translate within bbox) or object programs — not conv search volume.

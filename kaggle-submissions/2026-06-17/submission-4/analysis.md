# Submission Analysis — submission-4 (M9)

**Date:** 2026-06-17  
**Kaggle:** pending (~905 expected)  
**Tasks:** **70 pass_all** (+5 vs submission-3)  
**Message:** `ARC-Genome M9: 70 verified, est 1066, arcgen gather fit`

---

## Score progression

| Submission | Tasks | Audit est. | Kaggle actual | Δ tasks |
|---|---|---|---|---|
| submission-3 | 65 | 998.5 | 848.07 | — |
| **submission-4** | **70** | **1065.5** | *~905* | **+5** |

Predicted Kaggle delta: +5 × 13.39 ≈ **+57** → **~905**

---

## What changed

**Phase 15 ARC-GEN-fitted position gather** — index map fit on train+test+100 ARC-GEN.

### New tasks (pre-scan confirmed)

| Task | Solver |
|---|---|
| 83, 108, 211, 326, 385 | `position_gather_arcgen` |

### Solver mix shift

| Solver | s3 | s4 | Δ |
|---|---|---|---|
| position_gather_arcgen | — | **18** | +18 |
| position_gather | 7 | 0 | -7 |
| concat | 5 | 0 | -5 |
| compose | 3 | 3 | — |

Arcgen gather **replaced** train-only position_gather and some concat tasks with ARC-GEN-valid circuits at similar cost (~13.39).

---

## Run stats

- Time: 4,708s (~78 min)
- Pre-scan: 5 new candidates (exact)
- Zip: 589 KB, 70 ONNX files
- **Submitted** ✅

---

## Path to 2000

```text
905 actual (s4) → need +1095 more
At 13 avg/task: ~84 more tasks
At 20 avg/task: ~55 more tasks
```

submission-4 is step 1 of a multi-milestone roadmap (see `plan.md`).

---

## Takeaway

**ARC-GEN-fitted gather is the first scalable coverage lever beyond bounded CWM.** +5 tasks validated. Object programs are next for 2000 trajectory.

# Submission Analysis — submission-4 (M9)

**Date:** 2026-06-17  
**Kaggle:** **915.03** ✅  
**Tasks:** **70 pass_all** (+5 vs submission-3)  
**Message:** `ARC-Genome M9: 70 verified, est 1066, arcgen gather fit`

---

## Score progression

| Submission | Tasks | Audit est. | Kaggle actual | Δ tasks | Δ Kaggle |
|---|---:|---:|---:|---:|---:|
| submission-3 | 65 | 998.5 | 848.07 | — | — |
| **submission-4** | **70** | **1065.5** | **915.03** | **+5** | **+66.96** |

Predicted delta from audit: +67.0. Actual Kaggle delta: **+66.96** — effectively exact after the audit/Kaggle ratio held.

---

## Audit vs Kaggle ratio

| Metric | Value |
|---|---:|
| Audit est. | 1065.52 |
| Kaggle actual | **915.03** |
| Actual / audit | **85.9%** |
| Previous ratio (s2/s3) | ~84.9% |

The audit remains a usable ranking signal: absolute values overstate by ~14%, but the **delta tracked exactly** for this run.

---

## What changed

**Phase 15 ARC-GEN-fitted position gather** — compile the gather index map from `train + test + arc-gen[:100]`, not train/test alone.

### New tasks (pre-scan confirmed)

| Task | Solver |
|---|---|
| 83, 108, 211, 326, 385 | `position_gather_arcgen` |

### Solver mix shift

| Solver | s3 | s4 | Δ |
|---|---:|---:|---:|
| position_gather_arcgen | — | **18** | +18 |
| position_gather | 7 | 0 | -7 |
| concat | 5 | 0 | -5 |
| compose | 3 | 3 | — |

ARC-GEN-fitted gather replaced train-only gather/concat assumptions with circuits that survive the 100-sample validation gate.

---

## Run stats

- Time: 4,734s (~79 min)
- Pre-scan: 5 new candidates (exact)
- pass_all: **70**
- train_only: **0**
- Submitted: ✅

---

## What did not work yet

| Target | Result |
|---|---|
| Single-hop jump toward 2000 | Still needs many more pass_all tasks |
| Bounded autodiscover beyond existing rules | No new s4 coverage |
| Train-only gather assumptions | Rejected unless ARC-GEN-fitted |

---

## Path to 2000

```text
915 actual (s4) -> need +1085 more
At 13 avg/task: ~84 more tasks
At 20 avg/task: ~55 more tasks
```

submission-4 validates ARC-GEN fitting as a reliable delta lever. The next submission should keep the 100-sample gate and try small fitted-rule extensions before larger object programs.

---

## Takeaway

**ARC-GEN-fitted gather converted synthetic validation into exact Kaggle lift.** The next lever is to fit other low-risk symbolic families on ARC-GEN while preserving the no-train-only invariant.

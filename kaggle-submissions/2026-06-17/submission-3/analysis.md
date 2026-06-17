# Submission Analysis — submission-3 (M8b)

**Date:** 2026-06-17  
**Kaggle:** 848.07 ✅ (identical to submission-2)  
**Tasks:** 65 pass_all  
**Message:** `ARC-Genome M8b: 65 verified, est 999, autodiscover`

---

## vs submission-2

| Metric | submission-2 | submission-3 |
|---|---|---|
| pass_all | 65 | 65 |
| Audit est. | 998.5 | 998.5 |
| Kaggle actual | 848.07 | *pending (~848)* |
| New tasks | +1 (307) | 0 |
| Phase | 13 | **14** |

No coverage or score change expected — autodiscover infrastructure + regression fixes.

---

## What changed

1. **Autodiscover harness** (`autodiscover.py`) — 7 bounded rules, prescan, belief routing
2. **Seed baseline** — copy submission-2 ONNX before solve
3. **cost_audit preserve** — don't delete seeded ONNX on solve failure
4. **Bugfix** — `compile_dynamic_upscale2` tile stub NameError (broke task 307 in first run)
5. **third_pass disabled** — phase 14 saves ~30 min

---

## Pre-scan

```text
New pass_all candidates: 0
```

Autodiscover rules on 335 unsolved: **0 hits**. Library complete for single-step bbox transforms.

---

## First run regression (fixed)

First run: 64 tasks — task 307 deleted when upscale compile crashed. Fixed `bounded.py` + `cost_audit_task` seed preservation. Re-run: 65 tasks ✅

---

## Solver mix

Identical to submission-2 (65 tasks, same families).

---

## Takeaway

submission-3 is **infrastructure + stability**. Next score jump requires object programs or cheaper bounded compile — not autodiscover alone.

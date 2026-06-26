# Strategy — submission-2 (M10b)

**Goal:** Convert submission-1 flat probe into the next ARC-GEN-fitted symbolic family  
**Today's bet:** bbox-relative gather with dynamic index compiler

---

## Strategic framing

| Horizon | Target | Mechanism |
|---|---|---|
| submission-2 | +3–8 pass_all | bbox-relative ARC-GEN gather |
| M11 | 80+ pass_all | object extract/place programs |
| M12 | 100+ pass_all | multi-step Code World Models |

submission-1 proved Phase 16 is safe (70 preserved, 0 train_only) but not sufficient alone.

---

## AutoHarness v8

```text
1. set_phase(17)
2. seed submission-4 ONNX (same 70-task baseline)
3. pre-scan s_bbox_gather_arcgen on unsolved tasks
4. solve_all — bbox gather solvers before conv_diff
5. package validate_full pass_all ONNX only
6. audit phase 17
7. submit if pass_all > 70 OR est >= 1066.52, train_only == 0
```

---

## Guardrails

| Risk | Mitigation |
|---|---|
| Dynamic compiler bugs | Unit-test on s4 solved tasks first — must not regress |
| Bbox scalar mismatch | Reuse `_bbox_scalars` from bounded.py |
| Zero prescan hits | Skip submit; pivot to object programs in submission-3 |
| Regression vs s4 | Cost-audit + baseline task gate |

---

## Submit rule

```text
train_only == 0
AND
(pass_all > 70 OR audit_est >= 1065.52 + 1.0)
```

Seed from `kaggle-submissions/2026-06-17/submission-4/submission_v2.zip`.

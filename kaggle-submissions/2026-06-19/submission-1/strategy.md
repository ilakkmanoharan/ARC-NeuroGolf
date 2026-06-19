# Strategy — submission-1
Bootstrapped from kaggle-submissions/2026-06-17/submission-5 on 2026-06-19.

**Goal:** Continue toward 2000 without sacrificing the validated 70-task baseline  
**Today's bet:** ARC-GEN-fitted color maps

---

## Strategic framing

| Horizon | Target | Mechanism |
|---|---|---|
| submission-1 | Safe incremental probe | Fit color remaps on ARC-GEN |
| M10 | More symbolic families | Bbox-relative gather, object programs |
| M12 | 100+ pass_all | Multi-step Code World Models |

submission-1 should not chase train/test-only wins. It should either improve the strict audit or decline to submit.

---

## AutoHarness v7

```text
1. set_phase(16)
2. seed submission-4 ONNX from folder or submission_v2.zip
3. pre-scan color_map_arcgen on tasks not solved by submission-4
4. solve_all with cost_audit preservation
5. package only validate_full pass_all ONNX
6. audit with phase 16
7. submit only if pass_all increases OR est >= baseline + 1, with train_only == 0
```

---

## Guardrails

| Risk | Mitigation |
|---|---|
| No prior ONNX folder in repo | Restore from prior zip if present; otherwise solve from scratch |
| Train-only leakage | Submit gate requires `train_only == 0`; zip packaging validates full ARC-GEN |
| No new color-map tasks | Skip submit and use results to choose bbox/object next |
| Regression vs s4 | Baseline task and score gates block submit |

---

## Submit rule

Submit if and only if:

```text
train_only == 0
AND
(pass_all > 70 OR audit_est >= 1065.52 + 1.0)
```

This matches the automation rule: pass_all increase or estimated score +1 vs baseline.

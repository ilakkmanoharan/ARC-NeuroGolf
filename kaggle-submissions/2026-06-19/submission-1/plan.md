# Submission Plan — submission-1 (M10a)
Bootstrapped from kaggle-submissions/2026-06-17/submission-5 on 2026-06-19.

**Baseline:** 915.03 Kaggle, 70 pass_all  
**Audit baseline:** 1065.52  
**Goal:** preserve 70 and submit only on pass_all or estimated-score improvement

---

## Primary lever: ARC-GEN-fitted color maps (Phase 16)

submission-4 proved that small symbolic families become safer when their parameters are fitted on `train + test + arc-gen[:100]`.

For today's submission-1, extend that pattern to same-shape color remaps:

```text
input color -> output color
fit over train + test + 100 ARC-GEN
compile as 1x1 Conv
```

This is intentionally minimal:

- No new ONNX compiler
- Same 100-sample ARC-GEN validation gate
- No train_only packaging
- Cost-audit preservation of existing solved tasks

---

## Success criteria

| Metric | Target |
|---|---:|
| pass_all | > 70 |
| OR audit est. | >= 1066.52 |
| train_only | 0 |
| arcgen_validate_samples | >= 100 |

If neither improvement gate passes, the run should write results and skip Kaggle submit.

---

## Expected outcome

This is a low-risk probing submission. It may produce zero new tasks, but it establishes Phase 16 scaffolding for ARC-GEN-fitted symbolic families while preserving the submission-4 baseline behavior.

---

## Next if no submit

If color-map fitting yields no increase:

1. Add bbox-relative gather with a dynamic index compiler.
2. Add object-centric extract -> transform -> place programs.
3. Keep pre-scan and packaging tied to the same 100-sample gate.

# Learnings — submission-4 (M9)

**Outcome:** 70 pass_all, 915.03 Kaggle, +66.96 vs submission-3.

---

## Actionable rules

1. **Treat ARC-GEN as fitting evidence, not just validation.** The winning change moved parameter inference from train/test to train + test + 100 ARC-GEN.
2. **Never count train_only tasks as progress.** Local train/test wins only matter after the 100-sample ARC-GEN gate passes.
3. **Track deltas more than raw audit totals.** The audit absolute score was high by ~14%, but the s4 delta matched Kaggle almost exactly.
4. **Prefer low-risk symbolic family extensions.** Reusing the existing gather compiler kept the change small and made failures easy to reject.
5. **Seed preservation remains important.** Cost audit should keep prior ONNX when a new candidate is worse or absent.
6. **Pre-scan must match the production gate.** s4 worked because pre-scan and packaging both used full ARC-GEN validation.
7. **Large score goals require coverage, not only cheaper models.** At current yield, 2000 needs dozens more pass_all tasks.

---

## Carry-forward checklist

| Rule | Submission-5 requirement |
|---|---|
| 100 ARC-GEN samples | `set_phase(16)` must keep `arcgen_validate_samples >= 100` |
| No train_only | Submit only if audit reports `train_only == 0` |
| Real improvement | Submit only on `pass_all > 70` or audit estimate >= baseline + 1 |
| Minimal next lever | ARC-GEN-fit one cheap symbolic family before object-program expansion |

---

## Best next experiment

Start with **ARC-GEN-fitted color maps** because it requires no new ONNX compiler and extends the successful "fit symbolic parameters from generator samples" pattern. If it produces no pass_all increase, the next meaningful lever is bbox-relative gather or object-level programs.

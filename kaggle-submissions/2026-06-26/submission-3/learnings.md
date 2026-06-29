# Learnings — submission-3 (M11)

**Outcome:** 74 pass_all (local), +2 vs submission-2, Kaggle score pending.

---

## Actionable rules

1. **Variable-shape gravity needs dynamic ONNX, not static gather.** Tasks 32/78 prove the compiler path; do not revert to `_s_gravity_arcgen` for bbox-shifting grids.
2. **Targeted bundle builds are valid for +1–2 prescan hits** but must not skip the unsolved conv pass when prescan is otherwise empty.
3. **Prescan empty on 326 unsolved means symbolic families are saturated** — pivot to conv second-pass or deeper compose, not more single-step object rules.
4. **Seed preservation is mandatory.** Cost audit + seeded zip prevented regression on 72 baseline tasks.
5. **Submit when local gates pass** even if prior submission Kaggle logs are pending — north star uses latest *scored* baseline (915.03), and s3 clears it on audit projection.
6. **Wire all four dynamic gravity directions** before prescan — left/right were implemented but omitted from `ARCgen_GRAVITY_SOLVERS`.
7. **Do not burn 90 min on conv when prescan ≥ 1 on a cheap symbolic family** — s3 was the opposite case (prescan hit + skip conv).

---

## Carry-forward checklist

| Rule | submission-4 requirement |
|---|---|
| 100 ARC-GEN samples | `set_phase(22)` keeps `arcgen_validate_samples >= 100` |
| No train_only | Submit only if audit reports `train_only == 0` |
| Real improvement | `pass_all > 74` OR est >= **1116** |
| Full unsolved pass | Run `solve_all` on 326 unsolved with seeded baseline |

---

## Best next experiment

**Phase 22:** depth-3 compose + full seeded solve_all on unsolved pool. Expected 1–5 new pass_all from conv second-pass; compose depth-3 as secondary symbolic lever.

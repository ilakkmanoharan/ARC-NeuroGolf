# Learnings — submission-5 (M13)

**Outcome:** 74 kaggle-eligible (local), +2 vs submission-2 effective count; Kaggle public score **pending**.

---

## Actionable rules

1. **Always gate on `kaggle_eligible`, not raw `pass_all`.** submission-3 had 74 pass_all but only 72 counted — phantom est inflation.
2. **Dynamic gravity near-misses need slim compile, not deeper static compose.** Phase 21 lesson confirmed; slim variant (`build_gravity_model_slim`) is the production path.
3. **Targeted compile is valid when prescan hits are ≤2 and family-specific.** 102 s run vs 5 h conv pass with 0 gain (s4).
4. **Seed from last scored baseline zip (s2), not bloated s3 zip.** s3 ONNX for 32/78 was oversized; s2 had clean 72.
5. **Do not burn 90 min on conv when gravity prescan ≥ 1.** Opposite of s4 — run cheap symbolic first.
6. **Wait for public score before submission-6 implement.** Baseline moves if s5 lands ~965; cost-audit lever depends on confirmed task set.
7. **Size gate in submit path is mandatory.** Prevents repeat of s3 flat submit.

---

## Carry-forward checklist (submission-6)

| Rule | Requirement |
|---|---|
| Baseline | Use submission-5 zip after public score confirms |
| No train_only | Audit `train_only == 0` |
| Real improvement | `kaggle_eligible > prior` OR est delta > 1 pt with same count |
| Prescan before solve | ≥ 1 hit on unsolved OR explicit cost-audit scope |
| No oversized | `oversized == 0` always |

---

## Best next experiment

**Phase 24 cost audit re-solve:** seed 74 ONNX from s5, run `cost_audit_task` on top-10 highest `kaggle_cost` tasks (39, 57, 78, 32, 150, 155, 307). Expected +5–15 Kaggle at fixed pass_all if analytical replacements beat conv scores.

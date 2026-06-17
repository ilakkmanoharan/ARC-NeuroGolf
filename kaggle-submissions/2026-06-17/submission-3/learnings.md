# Learnings — submission-3 (M8b)

**Kaggle:** 848.07 | **Tasks:** 65 | **Δ vs submission-2:** 0

---

## 1. Identical score confirms autodiscover alone doesn't add coverage

submission-2 and submission-3 both scored **848.07** with 65 tasks. Autodiscover infrastructure is necessary but **not sufficient** — rules matched only on already-solved tasks.

---

## 2. Seed + cost_audit regression risk

First run dropped to **64 tasks** when `compile_dynamic_upscale2` threw NameError on tile stub — seeded task 307 ONNX was deleted. **Fix:** preserve seeded files when `cost_audit=True` and solve fails.

---

## 3. Audit ratio stable at 84.93%

| Submission | Est. | Actual | Ratio |
|---|---|---|---|
| submission-2 | 998.55 | 848.07 | 84.93% |
| submission-3 | 998.55 | 848.07 | 84.93% |

---

## 4. Path to 2000 requires coverage at scale

```text
848 / 65 ≈ 13.05 avg per task
2000 / 13.05 ≈ 153 tasks needed at current average
2000 / 25     ≈ 80 tasks at perfect score
```

**Implication:** score-only optimization cannot reach 2000. Need **~85–150 new pass_all tasks** depending on circuit cost.

---

## 5. Hidden gather opportunity discovered (post-analysis)

Train-only `position_gather` fails ARC-GEN, but **fitting indices on train+test+ARC-GEN** unlocks 5 tasks: 83, 108, 211, 326, 385 (~13.39 each → **~+57 Kaggle**).

This is the submission-4 bet.

---

## 6. Single-step bbox rules exhausted

Autodiscover library (flip, upscale, rot, transpose, scale_down): **0 hits on 335 unsolved**.

---

## 7. third_pass disabled saves ~30 min

No measurable coverage loss. Keep disabled until new conv representation emerges.

---

## One-line summary

> **submission-3 = stable infrastructure. Next win = ARC-GEN-fitted gather (+5 tasks), then object programs for 2000 path.**

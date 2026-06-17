# Learnings — submission-2 (M8)

**Kaggle:** 848.07 | **Tasks:** 65 | **Δ vs submission-1:** +12.95

---

## 1. Audit ratio is stable and predictable

| Submission | Audit est. | Kaggle actual | Ratio |
|---|---|---|---|
| submission-1 | 985.6 | 835.12 | 84.7% |
| **submission-2** | **998.5** | **848.07** | **84.9%** |

Delta prediction: +12.94 (task 307) → actual +12.95. **Exact.**

Use audit for ranking; use `est × 0.849` for Kaggle forecasting.

---

## 2. Coverage lever confirmed at scale

| Lever | submission-2 result |
|---|---|
| New bounded primitive (upscale2) | **+1 task, +12.95 Kaggle** |
| ARC-GEN routing alone | 0 (reordering only) |
| Third-pass conv | 0 wins (again) |

**Rule:** routing finds candidates; **new CWM primitives** capture them.

---

## 3. Variable-shape upscale was the gap

Task 307: shapes 2×2 → 4×4, 3×3 → 6×6, etc. Static `s_upscale` requires fixed input shape → unreachable. `bounded_upscale2` with runtime gh/gw solved it.

Pattern: when train examples have **varying input shapes** but consistent **bbox-relative rule**, use bounded dynamic compiler.

---

## 4. Bounded CWM is the score floor

Three bounded tasks score **12.9** (worst in zip). Memory 172–179k from gh/gw unroll + 900-cell gather.

40 conv tasks average ~15.3. **3× bounded tasks drag average by ~0.2 each.**

Cost optimization on bounded graphs is the highest-ROI path at constant 65 tasks.

---

## 5. Autodiscover reveals hidden structure

Post-hoc rule scan found bounded rules on **already-solved** tasks:

- rot180 on 87, 140 (using compose)
- transpose on 179, 241 (using analytical transpose at 25.0)
- rot90 on 380 (using compose)

But **0 rules on 335 unsolved** — remaining tasks need object programs or multi-step CWM, not single bbox transforms.

---

## 6. mask_preserve train-only trap persists

47 unsolved tasks pass mask_preserve on **raw grids** + ARC-GEN but fail on **onehot** semantics. Cannot include without correct channel-0 background handling.

---

## 7. Submission discipline validated again

+1 task justified submit (+12.95 Kaggle). submit-2 est 999 → actual 848 confirms forecasting.

---

## 8. Priority queue for submission-3+

| Priority | Action | Expected impact |
|---|---|---|
| **P0** | Autodiscover harness (systematic rule scan) | Foundation for future primitives |
| **P1** | Cheaper bounded compile (shared bbox) | +2–5 score at 65 tasks |
| **P2** | Object-centric programs | +10+ tasks (hard) |
| **P3** | Bounded rot/transpose compile | Only if cheaper than compose |
| **Avoid** | Re-submit identical zip | Wastes daily slot |
| **Avoid** | Third-pass conv | 0 wins proven |

---

## One-line summary

> **submission-2 proved bounded CWM + ARC-GEN routing unlocks variable-shape tasks (+12.95). Next: autodiscover pipeline + cheaper bounded graphs; coverage needs object programs.**

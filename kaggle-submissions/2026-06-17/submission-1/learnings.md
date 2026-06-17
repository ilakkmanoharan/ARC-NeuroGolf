# Learnings — submission-1 (M7b)

**Kaggle:** 835.12 | **Tasks:** 64 | **Δ vs M6:** +0.68

---

## 1. Two levers, two scales

| Lever | submission-1 result | Scale |
|---|---|---|
| **Coverage** (new pass_all tasks) | 0 new | +25 pts per task (M6 flip) |
| **Cost** (cheaper circuits, same tasks) | +0.68 | +0.34 per optimized task |

Cost tuning is real but small. Meaningful jumps require **new pass_all tasks**, not more search on the same 64.

---

## 2. Representation beats search volume

M3–M6 spent ~88 min/run with third-pass conv on 336 unsolved → **0 wins**.

What worked:
- M6: **dynamic bbox flip** (new representation)
- submission-1: **shared channel Gather** (compilation fix)

What didn't:
- Third-pass conv, depth-4 composition, M5 Gather primitives
- Bounded gravity (train+test only, ARC-GEN fails)

**Rule:** invest in expressiveness and compilation, not more passes.

---

## 3. ARC-GEN gate is non-negotiable

Task 32 (gravity_down) passes train+test but fails ARC-GEN example 6. Including it would be M1 all over again (199 solved → 23 earn).

**Rule:** never lower the 100-sample ARC-GEN gate to inflate zip size.

---

## 4. Onehot semantics matter for world models

Background color 0 encodes as **channel 0 = 1**, not all-zero. Bounded transforms must use:

```text
occupancy = max(channels 1..9) > 0.5   # not max(all channels)
bg fill   = within bbox only           # not full 30×30 padding
```

Getting this wrong made gravity look solvable when it wasn't.

---

## 5. Memory dominates score, not params

Bounded flip at submission-1:

| | params | memory | score |
|---|---|---|---|
| tasks 150/155 | 616 | 179,220 | 12.90 |

```text
score = max(1, 25 − ln(params + memory))
```

Optimizing params alone is insufficient. **Graph size → memory → score.**

Slim gather cut memory 29% (251k → 179k) and score +0.34/task. Still the worst-scoring tasks in the zip.

---

## 6. Audit estimates are consistently high

| Submission | Audit est. | Kaggle actual | Ratio |
|---|---|---|---|
| M6 | 984.9 | 834.44 | 84.7% |
| submission-1 | 985.6 | 835.12 | 84.7% |

Use audit for **ranking candidates**, not absolute score prediction. Trust official `kaggle_score_model` for selection.

---

## 7. Solver mix reveals the ceiling

64 tasks breakdown:
- **40 conv** (~13–16 pts each) — bulk of submission
- **22 analytical** (~17–22 pts each) — high value
- **2 bounded flip** (~12.9 pts each) — correct but expensive

Avg 13.05 pts/task. To reach 1,000 need either more tasks or higher average (~15.6 at 64 tasks).

---

## 8. Bounded CWM pattern (proven recipe)

```text
1. Identify shape-relative rule (flip, gravity, …)
2. Verify in numpy with correct onehot semantics
3. Verify on 100 ARC-GEN samples
4. Compile to dynamic ONNX (runtime gh/gw + Gather)
5. Slim compilation (shared subgraphs, single gather)
```

Flip passed all steps. Gravity failed step 3.

---

## 9. Submission discipline validated

submission-1 improved score with **same 64 tasks** — worth submitting (+0.68).

Don't submit identical zips (M3–M5 lesson). Do submit when official scorer confirms improvement.

---

## 10. Priority queue for submission-2+

| Priority | Action | Expected impact |
|---|---|---|
| **P0** | ARC-GEN hex → solver routing (M8) | +5–20 tasks |
| **P1** | Shared bbox subgraph (dedupe gh/gw unroll) | +1–3 score at 64 tasks |
| **P2** | Cost audit conv → analytical swap | +2–5 score at 64 tasks |
| **P3** | Object-centric programs (extract→transform) | +10+ tasks (harder) |
| **Avoid** | Gravity without generator routing | 0 pass_all |
| **Avoid** | Third-pass conv / more Gather primitives | 0 new wins proven |

---

## One-line summary

> **Coverage comes from new representations (bounded CWM, ARC-GEN routing); score per task comes from memory-efficient compilation. Search volume alone is saturated at 64.**

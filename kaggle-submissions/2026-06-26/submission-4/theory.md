# Theory — submission-4 (Phase 22 / M12)

**Hypothesis:** submission-3 left 326 tasks unsolved because it never ran the conv second-pass; symbolic families are saturated at depth-2 compose.

---

## Why conv second-pass should work

`solve_all` at phase 21+ enables:
- Full analytical + ARC-GEN solver stack per task
- **Second pass** with `unsolved_budget=90s` on remaining unsolved

submission-2's full solve found 72 tasks including 19 `conv_diff`, 14 `conv_fixed`. submission-3 only compiled 2 gravity tasks — **328 tasks never saw conv search**.

Historical pattern: conv finds train+test fits; ARC-GEN gate rejects train-only. Second-pass with official scoring (`use_official_score=True` @ phase 8+) filters unsafe wins.

---

## Why depth-3 compose

Depth-2 search covers ~18^2 = 324 two-op chains. Depth-3 adds chains like:

```text
color_map ∘ gravity_down ∘ flip_h
keep_3 ∘ largest_cc ∘ transpose
```

Some unsolved tasks may require three cheap primitives where two ops fail ARC-GEN.

Cost: slower per-task search; acceptable on 326 unsolved only.

---

## Why wire gravity left/right

Implementation exists (`s_gravity_left/right_dynamic`) but was omitted from solver list. Zero prescan cost; catches any horizontal gravity tasks missed by vertical-only wiring.

---

## Expected Kaggle delta

| Source | Δ pass_all | Δ est | Δ Kaggle (85.9% ratio) |
|---|---:|---:|---:|
| conv 2nd pass (1 task) | +1 | ~+15 | ~+13 |
| conv 2nd pass (3 tasks) | +3 | ~+45 | ~+39 |
| compose depth-3 (1 task) | +1 | ~+15 | ~+13 |

Conservative target: **75–78 pass_all**, est **1130–1160**, Kaggle **~970–996**.

---

## Risk

- Conv may find train-only fits → rejected by validate_full → 0 gain (same as prescan empty)
- Depth-3 compose may add ~10–20s per unsolved task → runtime within budget

---

## Phase config

```python
set_phase(22)  # phase 21 + arcgen_compose_depth=3
```

# Strategy — submission-3

**Informed by:** [submission-2/learnings.md](../submission-2/learnings.md), [kaggle_logs/README.md](../submission-2/kaggle_logs/README.md)

---

## Strategic shift

| | submission-2 | submission-3 |
|---|---|---|
| Bet | New bounded primitive (upscale2) | **Autodiscover infrastructure** |
| Coverage | +1 task (307) | 0 expected (prescan empty) |
| Score | +12.95 Kaggle | Maintain 848+ |
| Risk | Low | Low (seed baseline) |

submission-2 proved **one bounded primitive = +12.95**. submission-3 invests in **systematic discovery** because ad-hoc scans won't scale to 335 unsolved.

---

## Framework mapping

### AutoHarness v5

```text
1. prescan_new_tasks() — autodiscover on unsolved
2. seed submission-2 ONNX
3. solve with belief-ranked rules (phase 14)
4. cost_audit — monotonic score selection
5. submit if ≥ baseline
```

### BLF

Autodiscover rules provide **hard evidence** (100 ARC-GEN pass) not just signature priors. Confidence = 1.0 or skip.

### CWM

Rule library encodes bbox-relative semantics. Compilers only needed when prescan finds unsolved hits.

---

## Kaggle log insights (848.07)

```text
Delta vs s1:  +12.95 = task 307 score (12.94)  ← exact
Audit ratio:  0.8493 (stable)
Score floor:  bounded tasks at 12.9 (150, 155, 307)
Best tasks:   transpose at 25.0 (179, 241)
```

**Implication:** next coverage win needs rules **not in current library** (object extract/transform). Next score win needs **cheaper bounded compile**.

---

## Submit rules

| Outcome | Action |
|---|---|
| pass_all > 65 | Submit |
| pass_all = 65, est ≥ 998.4 | Submit (maintain) |
| pass_all < 65 | Do not submit |
| est drop > 1.0 | Do not submit |

---

## submission-4 preview

| Priority | Bet |
|---|---|
| P0 | Object-centric programs (extract → transform → compose) |
| P1 | Shared bbox subgraph (flip/upscale memory) |
| P2 | Bounded compiler for autodiscover hits |

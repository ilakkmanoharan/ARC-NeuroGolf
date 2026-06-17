# Strategy — submission-2

**Informed by:** [submission-1/learnings.md](../submission-1/learnings.md)

---

## Strategic shift

| | submission-1 | submission-2 |
|---|---|---|
| Bet | Cost optimization (slim flip) | **Coverage expansion (ARC-GEN routing)** |
| Lever | Compilation efficiency | **Belief-guided harness (BLF + AutoHarness)** |
| Expected Δ | +0.68 (proven) | +30–50 Kaggle (if 4–8 new tasks) |
| Risk | Low (monotonic) | Medium (routing may find 0 new) |

submission-1 validated the cost lever. submission-2 bets on the coverage lever because +0.68 per submission cannot reach 1,000.

---

## Framework mapping

### BLF (Bayesian Linguistic Forecaster)

```text
Prior:     P(solver family | arc_gen hex signature)
Evidence:  train+test I/O match per family
Update:    ARC-GEN 100-sample validation
Decision:  compile highest posterior that passes gate
```

Replaces blind solver enumeration order with **evidence-ranked search**.

### AutoHarness

```text
Harness v4:
  1. arcgen_route(task) → ordered hypothesis list
  2. bounded CWM solvers (belief order)
  3. analytical + extended + family
  4. composition
  5. conv (only if analytical fails)
  6. pass2/pass3 (unchanged, low expectation)
```

### Code World Models

Extend bounded CWM library only for families that pass ARC-GEN in numpy pre-scan. Do not compile gravity globally — route to gravity only when generator signature matches.

### ASRA

Generator-conditioned action semantics: "gravity_down" means different things per ARC-GEN family. Routing discovers which semantic context applies.

---

## Pre-submit checklist

- [ ] Numpy pre-scan: count tasks with new pass_all per family (min 4 to submit)
- [ ] No train-only inclusions (100 ARC-GEN gate on every zip file)
- [ ] Official score selection (`use_official_score=True`)
- [ ] Compare vs submission-1 baseline: tasks ≥ 64 AND score > 835.12
- [ ] Document in `submission-2/analysis.md` regardless of outcome

---

## Submit / no-submit rules

| Outcome | Action |
|---|---|
| pass_all > 64 | **Submit** |
| pass_all = 64, score > 835.12 | **Submit** (cost win) |
| pass_all = 64, score = 835.12 | **Do not submit** |
| pass_all < 64 | **Do not submit** (regression) |

---

## Risk mitigation

| Risk | Mitigation |
|---|---|
| Routing finds 0 new tasks | Pre-scan in numpy before 90-min solve; abort early if 0 |
| Generator metadata sparse | Fall back to train-signature clustering on arc_gen.json grids |
| Run time blowup | Cap bounded compile attempts per task; belief rank limits tries |
| Flip regression | Keep Phase 12 slim flip as default; routing adds, doesn't replace |

---

## Score math (why coverage matters)

```text
Current:  64 tasks × 13.05 avg = 835
Target:   68 tasks × 13.05 avg = 887   (+52 from 4 tasks alone)
Stretch:  72 tasks × 13.5  avg = 972
Goal:     80 tasks × 14    avg = 1,120
```

4 new tasks at current average beats another round of flip cost tuning.

---

## Kaggle budget

- **Limit:** 100/day
- **Used today:** 1 (submission-1)
- **Planned:** 1 (submission-2, conditional on pre-scan)

---

## Success definition

**Minimum viable:** 68 pass_all, Kaggle ≥ 870  
**Good:** 70+ pass_all, Kaggle ≥ 900  
**No submit:** unchanged 64 / 835.12

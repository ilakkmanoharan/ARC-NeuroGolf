# Strategy — submission-4

**Goal:** Increase Kaggle score toward **2000**  
**Today's bet:** ARC-GEN-fitted gather (+5 tasks)

---

## Strategic framing

| Horizon | Target | Mechanism |
|---|---|---|
| **Today (s4)** | ~905 Kaggle | Fit gather indices on 100 ARC-GEN samples |
| **M10** | ~1100–1300 | Object-centric programs |
| **M12** | ~2000 | 100+ pass_all at improved average |

2000 is not a single-submission goal — it's the **north star** requiring ~2.4× current coverage.

---

## BLF update

```text
Prior:    position_gather works on train (misleading)
Evidence: ARC-GEN disambiguates correct per-cell source mapping
Decision: compile gather with ARC-GEN-fitted idx
```

---

## AutoHarness v6

```text
1. prescan position_gather_arcgen on unsolved
2. seed submission-3 baseline
3. solve (arcgen gather solvers first)
4. cost_audit preserve
5. submit if pass_all > 65
```

---

## Risk

| Risk | Mitigation |
|---|---|
| arcgen gather picks worse score than compose on same task | Official score max selection |
| Only +5 not +50 | Document 2000 roadmap; M10 object programs |
| Seed regression | cost_audit preserve (from s3 fix) |

---

## Submit rules

Submit if **pass_all > 65** OR est improves by >1.0 at same task count.

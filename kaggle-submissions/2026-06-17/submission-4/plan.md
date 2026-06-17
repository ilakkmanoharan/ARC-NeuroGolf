# Submission Plan — submission-4 (M9)

**Baseline:** 848.07 Kaggle, 65 tasks  
**Stretch goal:** 2000 Kaggle  
**Realistic target:** **70 tasks, ~905 Kaggle**

---

## Score math (why 2000 is a multi-milestone goal)

```text
Current:  848 / 65  = 13.05 avg
Target:   2000

At current avg:  need 153 pass_all tasks (impossible in one jump)
At 20 avg:       need 100 pass_all tasks
At 25 avg:       need 80 pass_all tasks
```

**submission-4 delivers +5 tasks (+~57 Kaggle).** Path to 2000 = submission-5..N with object programs, dynamic multi-step CWM, conv→analytical swaps.

---

## Primary: ARC-GEN-fitted position gather (Phase 15)

### Discovery

12 tasks pass train-only `position_gather` but fail ARC-GEN. Fitting gather indices using **train+test+100 ARC-GEN** unlocks **5 tasks**:

| Task | Est. score |
|---|---|
| 83, 108, 211, 326, 385 | ~13.39 each |

### Implementation

| File | Change |
|---|---|
| `arcgen_gather.py` | `s_position_gather_arcgen`, `s_spatial_gather_arcgen` |
| `config.py` | Phase 15 `arcgen_fit_gather=True` |
| `infer.py` | Prioritize arcgen gather solvers |

---

## Success criteria

| Metric | Target |
|---|---|
| pass_all | **70** |
| Kaggle est. | **≥ 1065** (~905 actual) |
| Δ vs s3 | **+5 tasks, +57 Kaggle** |

Submit if pass_all > 65.

---

## Roadmap to 2000 (post submission-4)

| Phase | Lever | Est. tasks |
|---|---|---|
| M9 (s4) | ARC-GEN gather fit | +5 |
| M10 | Object programs (extract→rule→gather) | +20–40 |
| M11 | Cheaper bounded compile | +score at 70+ |
| M12 | Multi-step autodiscover chains | +30+ |

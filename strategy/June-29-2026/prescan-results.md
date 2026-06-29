# Prescan results — 2026-06-29

**Baseline:** submission-2 — 72 tasks, 940.75 Kaggle  
**Lane:** `2026-06-26/submission-3`  
**Phase:** 21 (`arcgen_dynamic_gravity`)

## Summary

| Scan | Unsolved hits |
|------|----------------|
| Phase 18 object | 0 |
| Phase 19 place | 0 |
| Phase 20 compose (depth 2–3) | 0 |
| **Phase 21 dynamic gravity** | **2** |

## New tasks

| Task | Hex | Solver | Rule |
|------|-----|--------|------|
| 32 | 1e0a9b12 | `gravity_down_dynamic` | `gravity_down`, variable 4–6² |
| 78 | 3906de3d | `gravity_up_dynamic` | `gravity_up`, fixed 10² |

Both pass `validate_full` (train + test + ARC-GEN).

## Still zero

- Compose confirms solved tasks 87, 140, 179, 380 only
- Conv sample (80 tasks): 0 hits
- Full analytical scan (55 solvers, pre-21): 0 hits

## Next step

```bash
# Cost audit on new models before solve_all
python3 -c "
from arc_genome.onnx.cost import compute_cost
from arc_genome.config import set_phase
set_phase(21)
# ... save + compute_cost for tasks 32, 78
"

NEUROGOLF_SKIP_KAGGLE_SUBMIT=1 python3 scripts/run_submission_2026-06-26_s3.py
```

Do **not** submit until cost audit shows positive estimated delta on the 74-task bundle.

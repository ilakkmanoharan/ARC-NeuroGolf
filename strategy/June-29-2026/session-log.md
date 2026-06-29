# Session log — June 29, 2026

## Context

Continuing from Phase 20 depth-2 compose (0 unsolved prescan hits). User approved dynamic gravity ONNX; asked to document under `strategy/June-29-2026/`.

## Timeline

### Morning — Phase 20 postmortem

- Depth-3 compose prescan: **0 hits** (confirmed depth-2 result).
- Full 55-solver prescan: **0 hits**.
- Near-miss analysis: tasks **32**, **78** match gravity in numpy; static gather fails due to **input-dependent column occupancy**.

### Phase 21 implementation

| Step | Result |
|------|--------|
| Numpy prototype (`colored` ≠ `in_grid`) | Tasks 32, 78 pass on onehot |
| ONNX v1 (full 30×30 unroll) | 4.6M nodes — too large |
| ONNX v2 (`max_grid_extent` bounding) | 52k–165k nodes |
| Slice batch dim fix | Task 78 partial |
| Boundary off-by-one fixes | Both tasks `validate_full=True` |
| `arcgen_gravity.py` + config level 21 | Wired in `infer.py`, run script |
| Prescan 328 unsolved | **+2 hits: 32, 78** |

## Files touched

```
arc_genome/onnx/gravity.py              # NEW — dynamic gravity compiler
arc_genome/genome/ops/arcgen_gravity.py # NEW — Phase 21 solvers
arc_genome/config.py                    # arcgen_dynamic_gravity @ 21
arc_genome/genome/infer.py              # solver priority
scripts/run_submission_2026-06-26_s3.py # Phase 21 prescan
strategy/June-29-2026/*                 # This documentation set
```

## Submission-3 status

| Gate | Status |
|------|--------|
| Prescan ≥ 1 unsolved | **PASS** (+2) |
| Cost audit | **TODO** |
| solve_all | Blocked on audit |
| Kaggle submit | Blocked on solve_all |

## Open items

- [ ] `compute_cost` / official score for tasks 32, 78
- [ ] Run submission-3 solve_all if audit positive
- [ ] Close submission-2 post-submit loop (`kaggle_logs`, GHA)
- [ ] Left/right dynamic gravity (if prescan warrants)

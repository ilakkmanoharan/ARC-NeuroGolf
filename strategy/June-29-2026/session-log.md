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

### Evening — submission-3 graded flat

| Observation | Detail |
|-------------|--------|
| Kaggle public | **940.75** (same as submission-2) |
| Local audit | 74 pass_all, est 1115 |
| Root cause | Tasks 32/78 ONNX **2.5 MB + 8.5 MB** &gt; **1.44 MB** Kaggle cap |
| Fix | `ONNX_MAX_BYTES` audit gate; submission-4 conv pass |

### Evening — submission-4 prep

- Phase 22 (depth-3 compose) already on `main`
- `run_submission_2026-06-26_s4.py` updated: seed without oversized, `kaggle_eligible` submit gate
- Docs: `score-flat-diagnosis.md`, `roadmap-2000.md`
- GHA: trigger `neurogolf-submission.yml` for submission-4

## Files touched

```
arc_genome/onnx/gravity.py
arc_genome/onnx/model.py                 # ONNX_MAX_BYTES
scripts/audit_submission.py              # oversized / kaggle_eligible tiers
scripts/run_submission_2026-06-26_s4.py  # size-aware seed + gates
strategy/June-29-2026/*
kaggle-submissions/2026-06-26/submission-3/analysis.md
```

## Open items

- [ ] GHA submission-4 completes (~90 min)
- [ ] Kaggle score &gt; 940.75 before celebrating
- [ ] Compress gravity compiler OR conv-solve 32/78 under 1.44 MB
- [ ] LoRA train on macos-14 GHA with size-gate examples

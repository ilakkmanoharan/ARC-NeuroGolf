# submission-3 notes

**Date:** 2026-06-29  
**Phase:** 21 — dynamic gravity ONNX

## Cost audit (74-task bundle)

| Metric | Value |
|--------|------:|
| pass_all | 74 |
| train_only | 0 |
| kaggle_score_est | 1114.99 |
| Δ vs submission-2 est | +23.76 |
| New tasks | 32 (`gravity_down_dynamic`), 78 (`gravity_up_dynamic`) |

## Build method

Seeded 72 ONNX from submission-2 + compiled gravity models for tasks 32/78.  
Skipped full 400-task `solve_all` (bundle audit passed; seeds unchanged).

## Submit

`kaggle_submit_ready.json` written — GHA `neurogolf-auto-submit` on push.

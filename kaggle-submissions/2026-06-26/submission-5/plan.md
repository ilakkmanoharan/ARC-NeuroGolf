# Submission Plan — submission-5 (M13)

**Baseline:** 72 kaggle-eligible, **940.75** Kaggle (submission-2)  
**Goal:** +2 Kaggle-counted tasks via **compact slim gravity** ONNX for 32/78

---

## Problem

submission-3 added tasks 32/78 with bloated dynamic gravity (2.5 MB / 8.5 MB). Kaggle rejected them → score stayed **940.75**.

## Fix

`build_gravity_model_slim` — single-Gather spatial index, background ch0 fill, compile-time `max_h×max_w` bounding.

| Task | Solver | Size | validate_full |
|---:|---|---:|---|
| 32 | `gravity_down_dynamic` | ~453 KB | ✅ |
| 78 | `gravity_up_dynamic` | ~889 KB | ✅ |

## Gates

- `kaggle_eligible > 72`
- `oversized == 0`
- `train_only == 0`
- est_eligible > submission-2 est + 1

Expected Kaggle delta: **+24–26** (tasks 32/78 official scores from prior audit).

## Prescan (326 unsolved)

Compact gravity: **2 hits** (32, 78 only). No other unsolved tasks match pure gravity rules.

## Also shipped

- Size gate in `infer._try_record` (reject oversized when `use_official_score`)
- `write_kaggle_submit_ready.py` blocks oversized submissions

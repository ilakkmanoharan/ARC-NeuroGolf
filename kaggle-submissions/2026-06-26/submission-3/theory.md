# Theory — submission-3 (Phase 21 / M11)

**Change:** Dynamic bbox-relative gravity ONNX  
**Result:** 74 pass_all (local), Kaggle pending

---

## BLF update

```text
Belief: static gather gravity fails when content bbox shifts across ARC-GEN samples.
Evidence: tasks 32/78 pass numpy gravity but fail static gather compile.
Decision: compile per-direction gravity with runtime gh/gw scalars + compile-time max extent.
```

---

## Problem

Phase 18 `_s_gravity_arcgen` and Milestone 5 `_s_gravity_*_var` infer gather indices from example 0. When ARC-GEN varies grid occupancy within the 30×30 canvas, the static index map reads wrong source cells.

Tasks 32 and 78 demonstrate pure vertical gravity rules that match train + test + 100 ARC-GEN but require **dynamic** source selection per column.

---

## Solution

`build_gravity_model(direction, max_h, max_w)` in `arc_genome/onnx/gravity.py`:

- Per-column (or per-row) colored-cell cumsum ranks
- Runtime `gh/gw` from `_bbox_scalars` mask inactive canvas region
- Compile-time loop bound = `max_grid_extent(task_data)` from train+test

This separates **rule verification** (numpy) from **graph compilation** (bounded loops at compile time, dynamic masking at runtime).

---

## Why only 2 tasks

Horizontal dynamic gravity (`left`/`right`) was implemented but not wired in submission-3. Prescan on 326 unsolved shows **zero** pure gravity rule matches beyond 32/78 — remaining unsolved tasks need multi-step transforms or conv fitting.

---

## Next lever

Run full solver stack (including conv second-pass) on unsolved tasks while preserving seeded pass_all ONNX. Extend compose search to depth-3 for chains like `color_map ∘ gravity_down`.

---

## Phase config

```python
set_phase(21)  # phase 20 + arcgen_dynamic_gravity
```

`arcgen_validate_samples` must remain **>= 100**.

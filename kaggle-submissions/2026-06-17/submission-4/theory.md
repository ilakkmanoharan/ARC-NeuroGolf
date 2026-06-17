# Theory — submission-4 (Phase 15 / M9)

**Change:** ARC-GEN-fitted gather index compilation

---

## Problem

`position_gather` infers `(sr,sc)` from train+test only. When ARC-GEN grids vary in content layout, train-consistent indices **fail held-out generators** — 12 unsolved tasks match train-only.

---

## Solution

Fit index map `idx[r,c] → (sr,sc)` such that **all** examples in `train + test + arc-gen[:100]` satisfy:

```text
output[r,c] = input[idx[r,c,0], idx[r,c,1]]
```

Compile to static `Gather` ONNX (same as analytical position_gather, different idx).

### Variable input shape

Fixed **output** shape required; input shape may vary (task 326). Index search uses bounds `GH×GW` with per-example bounds checks.

---

## Why this moves toward 2000

Each new pass_all task ≈ **+13 Kaggle** at current cost levels.  
5 tasks ≈ **+65 audit ≈ +55 Kaggle**.

Cumulative coverage path:

```text
65 → 70 (s4) → 90 (M10) → 120 (M12) → 150+ (2000 territory)
```

---

## Phase config

```python
set_phase(15)  # phase 14 + arcgen_fit_gather
```

---

## Expected outcome

| Metric | Value |
|---|---|
| pass_all | 70 |
| Kaggle | ~905 |
| New solver | `position_gather_arcgen` |

# Theory — submission-4 (Phase 15 / M9)

**Change:** ARC-GEN-fitted gather index compilation  
**Result:** 70 pass_all, 915.03 Kaggle

---

## BLF update

```text
Belief: train/test-consistent gather maps are underdetermined.
Evidence: ARC-GEN samples disambiguate source indices.
Decision: fit symbolic parameters on train + test + 100 ARC-GEN.
```

The useful abstraction is not "more search"; it is **parameter fitting under synthetic generator evidence**.

---

## Problem

`position_gather` inferred `(sr, sc)` from train/test only. That creates value-ambiguous maps: several source cells can match tiny train/test grids, but only one survives generator variation.

Train-only candidates are not useful for the competition loop because they inflate local coverage without Kaggle lift.

---

## Solution

Fit index map `idx[r,c] -> (sr,sc)` such that every example in `train + test + arc-gen[:100]` satisfies:

```text
output[r,c] = input[idx[r,c,0], idx[r,c,1]]
```

Then compile the result to the existing static `Gather` ONNX graph. The model class stays cheap; only the fitting evidence changes.

### Why it generalized

- The generated samples varied enough to remove ambiguous source cells.
- Validation used the same 100-sample gate before packaging.
- The audit delta and Kaggle delta matched almost exactly.

---

## Code World Model implication

The successful CWM pattern is:

```text
symbolic family -> ARC-GEN parameter fit -> ONNX compile -> 100-sample validate
```

This is a safer path than adding train-only heuristics. New levers should extend the set of symbolic families that can be ARC-GEN-fitted.

---

## Next lever

Phase 16 should stay small: fit another low-risk symbolic family on ARC-GEN before larger object-program work.

Candidate order:

| Lever | Reason |
|---|---|
| ARC-GEN-fitted color maps | Cheap, same-shape, no new compiler |
| Bbox-relative gather | Plausible for layout variation, needs dynamic index compiler |
| Object programs | Larger coverage potential, higher implementation risk |

---

## Phase config

```python
set_phase(15)  # phase 14 + arcgen_fit_gather
```

For follow-up work, `arcgen_validate_samples` must remain **>= 100** and train_only outputs must never be packaged.

---

## Takeaway

**ARC-GEN is most valuable as a parameter-fitting oracle for symbolic Code World Models.** The submission-4 lift proves that fitted parameters can translate directly to Kaggle score when the validation gate is strict.

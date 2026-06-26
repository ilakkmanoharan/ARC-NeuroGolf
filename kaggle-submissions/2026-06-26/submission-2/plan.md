# Submission Plan — submission-2 (M10b)

**Baseline:** 915.03 Kaggle, 70 pass_all (submission-4)  
**Local probe:** submission-1 Phase 16 — 70 pass_all, est 1066, **0 new tasks**  
**Goal:** break 70-task ceiling with layout-relative ARC-GEN fitting

---

## Primary lever: bbox-relative ARC-GEN gather (Phase 17)

submission-1 confirmed color_map_arcgen preserves the s4 mix (4 color_map_arcgen tasks) but adds **zero** new pass_all tasks. Pre-scan was empty.

Next symbolic family: **fit gather indices relative to runtime content bbox**, not absolute grid coordinates.

```text
for each output cell (r,c) in train + test + arc-gen[:100]:
  sr, sc = relative offset within input content bbox
compile dynamic Gather with bbox scalars (reuse bounded.py pattern)
```

---

## Implementation scope

| Component | File | Change |
|---|---|---|
| Relative index fitter | `arc_genome/genome/ops/arcgen_gather.py` | `s_bbox_gather_arcgen(td)` |
| Dynamic ONNX compiler | `arc_genome/onnx/bounded.py` | `compile_bbox_relative_gather()` |
| Harness | `arc_genome/genome/infer.py` | Phase 17 priority before conv search |
| Config | `arc_genome/config.py` | Phase 17 `arcgen_fit_bbox_gather` |

---

## Success criteria

| Metric | Target |
|---|---:|
| pass_all | **> 70** |
| OR audit est. | >= **1066.52** |
| train_only | 0 |
| prescan_new | >= 1 |
| arcgen_validate_samples | >= 100 |

Submit only if gates pass; otherwise document prescan misses for object-program pivot.

---

## Pre-scan target

Run `s_bbox_gather_arcgen` on all 330 unsolved tasks before the 90-min solve. Expect 3–8 candidates if absolute gather misses bbox-relative layouts.

---

## Fallback if flat

Object-centric extract → transform → place (Milestone 5 primitives + ARC-GEN gate).

# Theory — submission-2 (Phase 17 / M10b)

**Change:** ARC-GEN-fitted bbox-relative gather indices

---

## Hypothesis

Some tasks pass train-only `position_gather` but fail when ARC-GEN samples shift the content bbox. Absolute index maps `(r,c) → (sr,sc)` break when the foreground region moves.

Fitting **offsets relative to runtime bbox** `(r,c) → (bbox_r + dr, bbox_c + dc)` should generalize like submission-4's arcgen gather fix, but for layout-relative transforms.

---

## Evidence from submission-1 (local, 2026-06-26)

| Metric | s4 baseline | s1 Phase 16 |
|---|---:|---:|
| pass_all | 70 | 70 |
| est | 1065.52 | 1065.52 |
| prescan new | 5 (gather) | **0** (color) |
| color_map_arcgen | — | 4 (replacements, not new) |

Color remaps were the smallest extension; they did not expand coverage. Layout-relative gather is the next incremental symbolic family per submission-4 learnings.

---

## Model family

```text
bbox = content bounding box of input (dynamic at runtime)
for each output cell: output[r,c] = input[bbox_r + dr, bbox_c + dc]
fit (dr, dc) or full relative idx over train + test + arc-gen[:100]
compile as Gather + bbox scalar nodes (Phase 11 bounded world pattern)
```

Reject if:
- inconsistent relative map across examples
- ONNX fails 100-sample validation
- train_only packaging would occur

---

## Expected outcome

Moderate lift (+3–8 tasks) if bbox-relative layouts explain a slice of the 330 unsolved pool. Zero lift is acceptable if prescan is empty — confirms need for object programs.

---

## Why not skip to object programs

Object extract/place needs more compiler surface area. Bbox-relative gather reuses existing Gather + bounded scalar infrastructure with minimal new ONNX risk — same pattern that delivered +5 tasks in submission-4.

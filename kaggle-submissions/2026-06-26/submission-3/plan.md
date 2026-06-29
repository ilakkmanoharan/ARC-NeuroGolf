# Submission Plan — submission-3 (M11)

**Baseline:** 940.75 Kaggle, 72 pass_all (submission-2)  
**Goal:** break 72-task ceiling with ARC-GEN-fitted object programs

---

## Pre-scan result (2026-06-29, updated)

**0 new hits** on 328 unsolved after Phase 19 place programs:
`bbox_shift`, `patch_paste`, `translate`, `pad_embed`, `scale_down`, `varying_origin_paste`, object programs.

`patch_paste_arcgen` confirms **task 87** (already solved). Cost audit on 72 baseline: **+0 est**.

Do **not** run full `solve_all` or submit until prescan ≥ 1 on unsolved.

---

## Pivot: multi-step composition (M12)

Single-step place/object exhausted. Next compiler target:

```text
depth-2 compose: bounded_flip ∘ bbox_gather, gravity ∘ color_map, etc.
ARC-GEN gate each chain; prescan before solve
```

**Code target:** extend `arc_genome/genome/compose/search.py` with place-aware ops

---

## Wired (ready when prescan hits): Phase 19 place programs

submission-2 delivered bbox-relative gather (+2 tasks). Next family per roadmap: **rule transform + gather compile**, fitted on train + test + ARC-GEN[:100].

```text
gravity (4 dirs) | largest_cc | hollow_rect | fill_rect | keep/remove color
→ verify rule on all ARC-GEN examples
→ compile spatial/position gather ONNX
```

**Code:** `arc_genome/genome/ops/arcgen_object.py`, Phase 18 in `config.py`, priority in `infer.py`.

---

## Implementation scope

| Component | File | Change |
|---|---|---|
| ARC-GEN object fitters | `arc_genome/genome/ops/arcgen_object.py` | `ARCgen_OBJECT_SOLVERS` |
| Phase flag | `arc_genome/config.py` | `arcgen_fit_object_programs` @ level 18 |
| Harness | `arc_genome/genome/infer.py` | Object solvers before gather stack |
| Run script | `scripts/run_submission_2026-06-26_s3.py` | prescan + solve_all + gates |

---

## Success criteria

| Metric | Target |
|---|---:|
| pass_all | **> 72** |
| OR audit est. | >= **1092.2** |
| train_only | 0 |
| prescan_new | >= 1 |
| arcgen_validate_samples | >= 100 |

---

## Pre-scan

Run all `ARCgen_OBJECT_SOLVERS` on 328 unsolved tasks. Expect 3–10 hits if object-motion/selection explains a slice of the pool.

---

## Fallback if flat

Multi-step composition (depth-2) or conv → analytical swap per `strategy/unsolved-tasks-roadmap.md`.

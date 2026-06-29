# Submission Plan — submission-4 (M12)

**Baseline:** 74 pass_all, est 1115 (submission-3) | 915.03 Kaggle north-star  
**Goal:** break 74-task ceiling via full unsolved solve + depth-3 compose

---

## NeuroGolf-Strategize

**Question:** Given diagnosis, what single lever most likely beats 915.03 / 74 pass_all?

**Answer:** **Full seeded `solve_all` on 326 unsolved tasks** with Phase 22 (depth-3 compose + all dynamic gravity directions). submission-3 skipped conv second-pass; symbolic prescan is 0 across gather/object/place/compose-2. Conv budget 25 + second-pass 90s is the highest-probability path to +1–5 pass_all.

**Expected pass_all delta:** +1 to +5 (conservative), up to +8 if conv finds multiple ARC-GEN-safe fits.

---

## Primary lever: unsolved conv pass (Phase 22)

```text
1. Seed 74 ONNX from submission-3/submission_v2.zip
2. Prescan depth-3 compose + 4-way dynamic gravity on 326 unsolved
3. solve_all(task_nums=unsolved, conv_budget=25)  # ~60–75 min
4. Audit full submission; gate pass_all > 74 OR est >= 1116
```

---

## Secondary lever: compose depth-3

Phase 20 depth-2 returned 0 prescan on unsolved. Phase 22 bumps `arcgen_compose_depth` to 3 and wires `gravity_left/right_dynamic`.

Chains like `color_map ∘ gravity_down`, `flip_h ∘ largest_cc` may match tasks where depth-2 failed.

---

## Implementation scope

| Component | File | Change |
|---|---|---|
| Phase 22 flag | `arc_genome/config.py` | `arcgen_compose_depth = 3` @ level 22 |
| Gravity wiring | `arc_genome/genome/ops/arcgen_gravity.py` | add left/right to `ARCgen_GRAVITY_SOLVERS` |
| Run script | `scripts/run_submission_2026-06-26_s4.py` | seed s3 + unsolved solve_all + gates |

---

## Success criteria

| Metric | Target |
|---|---:|
| pass_all | **> 74** |
| OR audit est. | >= **1116.0** |
| train_only | 0 |
| prescan_new | >= 0 (conv may win without prescan) |
| arcgen_validate_samples | >= 100 |

Submit only if gates pass vs submission-3 baseline.

---

## Fallback if flat

Document in `notes.md`; do **not** write `kaggle_submit_ready.json`. Next pivot: conv→analytical swap on high-cost seeded tasks (score lift without new tasks) or depth-4 compose with pruned search.

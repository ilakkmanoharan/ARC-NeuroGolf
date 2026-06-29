# Why the Kaggle score stayed at 940.75

**Date:** 2026-06-29  
**Submissions compared:** submission-2 (72 ONNX) vs submission-3 (74 ONNX, M11 dynamic gravity)

---

## Symptom

All three recent Kaggle submissions show **public score 940.75**:

| Submission | Description | Local audit | Kaggle public |
|------------|-------------|------------:|--------------:|
| submission-2 | M10b, 72 verified | est ~1091 | **940.75** |
| submission-3 (GHA) | M11, 74 verified, dynamic gravity | est ~1115 | **940.75** |
| submission-3 (local dup) | same zip | est ~1115 | **940.75** |

Local `pass_all` went **72 → 74** (+tasks 32, 78). Kaggle score did **not** move.

---

## Root cause: ONNX file size limit (confirmed)

Competition rule ([Kaggle / CompeteHub](https://www.competehub.dev/en/competitions/kaggleneurogolf-2026)):

> Each ONNX file is limited to at most **1.44 MB**.

| Task | `validate_full` (local) | File size | Kaggle counts? |
|------|-------------------------|----------:|:--------------:|
| 32 | True | **2.50 MB** | **No** — over limit |
| 78 | True | **8.50 MB** | **No** — over limit |
| All 72 in submission-2 | True | all &lt; 1.44 MB | Yes |

submission-3 zip diff vs submission-2: **only** `task032.onnx` and `task078.onnx` were added. Both exceed 1.44 MB. Kaggle rejects oversized files, so the effective bundle is still **72 tasks** → **940.75**.

This is **not** primarily an ARC-GEN private-benchmark mismatch. Phase 21 dynamic gravity compiles correctly in our harness but **unrolls too many nodes** into the protobuf (165k+ nodes for task 78), blowing the size cap.

---

## Why local audit lied

`scripts/audit_submission.py` checked:

1. Train/test correctness  
2. ARC-GEN validation (`validate_full`)  
3. Official cost formula (`kaggle_score_model`)

It did **not** check **file size ≤ 1.44 MB**. Tasks 32/78 passed all three tiers locally but would never score on Kaggle.

**Audit ratio (stable):**

```text
940.75 / 1091.23 ≈ 86.2%   (submission-2 est → actual)
940.75 / 1114.99 ≈ 84.4%   (submission-3 est → actual, same Kaggle score)
```

The est bump (+24) was phantom — those two tasks never entered the official scorer.

---

## Phase 21 gravity — correct logic, wrong packaging

Dynamic gravity (column cumsum + rank matching) is the right algorithm for tasks 32/78. The compiler in `arc_genome/onnx/gravity.py` bounds graph size with `max_grid_extent()` but still emits per-cell Slice/ReduceSum trees that explode protobuf size:

- Task 32 (max 6×6): 2.5 MB  
- Task 78 (max 10×10): 8.5 MB  

**Lesson:** `validate_full=True` is necessary but not sufficient. Every submit candidate must pass **`os.path.getsize(path) <= 1_509_949`** (1.44 × 1024²).

---

## Fixes (ordered)

| Priority | Action | Expected effect |
|----------|--------|-----------------|
| P0 | Add `onnx_max_bytes` gate to audit + submit scripts | Stop phantom est inflation |
| P0 | Drop oversized seeds; re-solve 32/78 via **conv** (small ONNX) | +2 Kaggle tasks if conv fits |
| P1 | submission-4: full **conv second-pass** on 326 unsolved | +1–10 tasks (rogermt: conv solves ~220) |
| P2 | Compress gravity compiler (shared subgraphs, fewer Slice nodes) | Analytical path for gravity without conv |
| P3 | Cost audit swaps on existing 72 | Score lift without new coverage |

---

## Do not repeat

- Submit when `pass_all` increases but **any** file &gt; 1.44 MB  
- Trust est score without Kaggle-size gate  
- Seed submission-4 from submission-3 zip without stripping tasks 32/78  

---

## References

- [rogermt/ARC-AGI](https://huggingface.co/rogermt/ARC-AGI) — 276/400 tasks, 4106 total (~14.9/task); one-hot conv dominates  
- [ashhhhhh26/neurogolf-2026](https://huggingface.co/ashhhhhh26/neurogolf-2026) — analytical-first, conv_budget tuning  
- Prior lane: `kaggle-submissions/2026-06-17/submission-3/learnings.md` — identical score when zip adds 0 Kaggle-counted tasks

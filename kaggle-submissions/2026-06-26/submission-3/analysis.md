# Submission Analysis — submission-3 (M11)

**Date:** 2026-06-26  
**Kaggle:** **940.75** (unchanged vs submission-2) — submitted 2026-06-29  
**Tasks:** **74 pass_all** local (+2 vs submission-2) | **72 Kaggle-effective**  
**Audit est.:** **1115.0** (+23.8 vs s2 est 1091.2) — **phantom** (+2 oversized ONNX)  
**Message:** `ARC-Genome M11: 74 verified, est 1115, dynamic gravity`

---

## NeuroGolf-Diagnose

**Question:** How can we improve Kaggle score vs **915.03 / 70 pass_all** (north-star baseline) or **940.75 / 72** (submission-2 local)?

**Answer:** submission-3 **did not** improve Kaggle score. Tasks 32/78 pass `validate_full` locally but ONNX files exceed **1.44 MB** (2.5 MB + 8.5 MB) — Kaggle rejects them. Effective count stays **72 / 940.75**. Next: conv second-pass on unsolved + audit size gate.

---

## Kaggle flat score — root cause

| Task | `validate_full` | File size | Kaggle counts? |
|------|-----------------|----------:|:--------------:|
| 32 | True | 2.50 MB | **No** |
| 78 | True | 8.50 MB | **No** |

See `strategy/June-29-2026/score-flat-diagnosis.md`.

| Submission | pass_all | Audit est. | Kaggle actual | Δ tasks | Δ est |
|---|---:|---:|---:|---:|---:|
| submission-2 | 72 | 1091.2 | *pending* | — | — |
| north-star (s4 2026-06-17) | 70 | 1065.5 | **915.03** | — | — |
| **submission-3** | **74** | **1115.0** | **940.75** | **+0 Kaggle** | phantom +23.8 est |

Audit/Kaggle ratio: **940.75 / 1091.2 ≈ 86.2%** (submission-2). submission-3 est inflation from oversized tasks.

---

## What changed

**Phase 21 dynamic gravity ONNX** — compile-time bounded graphs with runtime bbox scalars; fixes static gather failures on variable-shape gravity (tasks 32, 78).

| Task | Solver | Notes |
|---|---|---|
| 32 | `gravity_down_dynamic` | 6×6 max extent, ~52k nodes, **2.5 MB** — official score **12.31** if accepted |
| 78 | `gravity_up_dynamic` | 10×10 max extent, ~165k nodes, **8.5 MB** — official score **11.46** if accepted |

Build method: seeded 72 ONNX from submission-2 + compiled gravity for 32/78. **No full 400-task `solve_all`** — bundle audit only.

---

## What blocked further gain

| Lever | Prescan on 326 unsolved | Blocker |
|---|---|---|
| compose depth-2 | 0 | chains exhausted at depth 2 |
| object programs | 0 | fixed-shape gather; variable bbox fails |
| place programs | 0 | no ARC-GEN pass on unsolved |
| bbox/position gather | 0 | already captured in s1–s2 |
| gravity left/right dynamic | 0 | no pure horizontal gravity tasks in pool |
| numpy gravity rules | 0 | no unsolved task matches pure gravity |

**Root cause:** submission-3 shipped a targeted 2-task compile without running conv second-pass on the 326-task unsolved pool.

---

## Ranked levers (expected Kaggle delta)

| Rank | Lever | Expected Δ pass_all | Expected Δ Kaggle | Confidence |
|---:|---|---:|---:|---|
| 1 | **Full seeded solve_all on unsolved** (conv 2nd pass) | 1–5 | +15–75 | medium |
| 2 | **Phase 22 compose depth-3** | 0–3 | 0–45 | low |
| 3 | gravity left/right dynamic wiring | 0–1 | 0–15 | low |
| 4 | conv → analytical swap (cost) | 0 | +5–20 score | low |

---

## Run stats

- Seeded: 72 from submission-2
- New: tasks 32, 78
- train_only: **0**
- Submitted: ✅ (GHA auto-submit 2026-06-29)

---

## Takeaway

Dynamic gravity validated the variable-shape compiler path (+2 pass_all). **The next submission must run full `solve_all` on the unsolved pool** — symbolic prescan is exhausted; conv second-pass is the highest-probability remaining lever.

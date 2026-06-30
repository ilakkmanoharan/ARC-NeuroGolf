# Submission Analysis — submission-5 (M13)

**Date:** 2026-06-26  
**Kaggle public:** **pending** (submitted 2026-06-30 via GHA)  
**Prior scored baseline:** **940.75** / **72** kaggle-eligible (submission-2)  
**Local:** **74 pass_all** | **74 kaggle-eligible** | **0 oversized**  
**Audit est.:** **1116** (+24.7 vs s2 est 1091.2)  
**Message:** `ARC-Genome M13: 74 verified, est 1116, compact gravity`

---

## NeuroGolf-Diagnose

**Question:** How can we improve Kaggle score vs **940.75 / 72** pass_all (submission-2 scored baseline)?

**Answer (provisional):** submission-5 fixes the submission-3 **size rejection** on tasks 32/78. Compact slim gravity ONNX (453 KB / 889 KB vs 2.5 MB / 8.5 MB) should restore **+2 Kaggle-counted tasks** (~**+24–26** public score if audit ratio holds). Public grade pending GHA post-submit. Next lever after confirmation: **cost_audit re-solve** on high-cost conv/gather tasks for score lift at fixed 74 pass_all.

---

## submission-3 vs submission-5 — same tasks, different packaging

| Metric | submission-3 | submission-5 | Δ |
|---|---:|---:|---:|
| pass_all (local) | 74 | 74 | 0 |
| kaggle_eligible | **72** | **74** | **+2** |
| oversized | **2** (32, 78) | **0** | fixed |
| Task 32 size | 2.50 MB | **453 KB** | ✅ |
| Task 78 size | 8.50 MB | **889 KB** | ✅ |
| Kaggle actual | **940.75** (flat) | pending | — |
| Audit est. | 1115.0 | **1116** | +1 |

submission-3 proved dynamic gravity compiles and validates; submission-5 proves **size-safe slim compile** so Kaggle accepts the circuits.

---

## Per-task audit (new tasks)

| Task | Solver | ONNX | kaggle_score (est) | pass_all |
|---:|---|---:|---:|:---:|
| 32 | `gravity_down_dynamic` | 453 KB | 12.40 | ✅ |
| 78 | `gravity_up_dynamic` | 889 KB | 12.28 | ✅ |

Both under **1.44 MB** gate; both pass `validate_full` (train + test + ARC-GEN[:100]).

---

## Ranked levers (expected Kaggle delta)

| Rank | Lever | Expected Δ pass_all | Expected Δ Kaggle | Confidence |
|---:|---|---:|---:|---|
| 1 | **Await s5 public score** — confirm +2 tasks land | 0 | +24–26 | high |
| 2 | **Cost audit swap** on high-cost tasks (39, 57, 150, 155, 307) | 0 | +5–15 | medium |
| 3 | **flip ∘ gravity compose** prescan on unsolved | 0–2 | 0–30 | low |
| 4 | Full unsolved conv pass (Phase 22) | 0 | 0 | low (s4 proved 0) |

---

## Run stats

- Seeded: 72 size-safe ONNX from submission-2
- New compiled: tasks 32, 78 (compact gravity)
- Prescan on 326 unsolved: **2 hits** (32, 78 only — gravity family saturated)
- train_only: **0**
- elapsed: **102 s** (targeted compile, no full solve_all)
- Submitted: ✅ (GHA auto-submit 2026-06-30)

---

## Takeaway

submission-5 is the **correct packaging fix** for submission-3's phantom +2. Do not re-run full conv solve_all until public score confirms the lift. If score lands ~965, pivot to **cost optimization** and **compose+gravity** chains on the remaining 326 unsolved pool.

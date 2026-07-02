# Submission Plan — submission-6 (M14)

**Baseline:** **965.42** Kaggle, **74** kaggle-eligible (submission-5)  
**Audit baseline:** **1116** est  
**Goal:** score lift via cost_audit + prescan-guided coverage (no 5h blind conv)

---

## Primary: Phase 24 cost_audit rescoring

Re-solve high-cost tasks with `cost_audit_task` (monotonic official score):

| Task | Why |
|---:|---|
| 39, 57, 150, 155, 307 | conv/gather — likely cheaper analytical swap |

Expected: **0 new pass_all**, **+5–15 Kaggle** if swaps land.

---

## Secondary: prescan-guided unsolved pass

1. `autodiscover` rules (compose, flip ∘ gravity, …)  
2. gravity prescan (saturated but cheap)

**Only** run conv on prescan hits. Skip full unsolved conv (submission-4 proved 0 gain in 5.5h).

---

## Success criteria

| Metric | Target |
|---|---:|
| kaggle_eligible | > 74 **OR** est_eligible ≥ 1117 |
| oversized | 0 |
| train_only | 0 |

---

## Continual harness

Post-submit queues this run via `scripts/queue_next_submission.py` — no laptop, no Cursor API required.

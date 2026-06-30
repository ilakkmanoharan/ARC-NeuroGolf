# Agent state — read at step 0

Last updated: 2026-06-30 — submission-5 submitted; awaiting public score.

## Baseline (latest scored)

| Field | Value |
|-------|------:|
| Folder | `kaggle-submissions/2026-06-26/submission-2` |
| Kaggle public | **940.75** |
| kaggle_eligible | **72** |

## Active submit

| Field | Value |
|-------|-------|
| Folder | `kaggle-submissions/2026-06-26/submission-5` |
| Phase | **23** compact slim gravity |
| kaggle_eligible | **74** (+2 vs s2) |
| est_eligible | **1116** |
| oversized | **0** |
| Status | **submitted** 2026-06-30 — public score **pending** GHA post-submit |

## What changed

- `build_gravity_model_slim` — tasks 32/78 under 1.44 MB, pass ARC-GEN
- Size gate in `infer._try_record` + `write_kaggle_submit_ready.py`
- Local kaggle_logs built; diagnosis committed (analysis/theory/learnings)

## Expected Kaggle

~**965** (+24 vs 940.75) if audit ratio holds on +2 tasks.

## Next agent action

1. Wait for `ci: post-submit logs submission-5` commit with `kaggle_score_actual`
2. Re-diagnose if public score differs from ~965 projection
3. Strategize submission-6: **Phase 24 cost_audit re-solve** on high-cost tasks (39, 57, 150, 155, 307)

## Do not repeat

- Submit bloated dynamic gravity (>1.44 MB)
- Count pass_all without kaggle_eligible gate
- Full 5h conv pass when prescan empty (s4 proved 0 gain)
- Implement submission-6 before submission-5 public score lands

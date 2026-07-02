# Agent state — read at step 0

Last updated: 2026-07-01 — submission-5 scored **965.42**; submission-6 queued.

## Baseline (latest scored)

| Field | Value |
|-------|------:|
| Folder | `kaggle-submissions/2026-06-26/submission-5` |
| Kaggle public | **965.42** |
| kaggle_eligible | **74** |

## Active work

| Field | Value |
|-------|-------|
| Folder | `kaggle-submissions/2026-06-26/submission-6` |
| Phase | **24** cost_audit + prescan-guided unsolved |
| Run script | `scripts/run_submission_2026-06-26_s6.py` |
| Status | **queued** via continual harness |

## What changed (submission-5)

- Compact slim gravity — tasks 32/78 under 1.44 MB
- **+24.67** Kaggle vs 940.75 (confirmed public score)
- 74 kaggle_eligible, 0 oversized

## Next agent action

1. Let GHA run submission-6 (cost_audit on 39,57,150,155,307 + prescan)
2. If flat: add compose prescan families or rogermt-scale conv on prescan-positive subset only
3. Export LoRA rows from s5 analysis → push examples → GHA train

## Do not repeat

- Full 5h unsolved conv when prescan empty (s4)
- Stale `work_dir` on old plan.md folders (fixed in submission_lane.py)
- Wait for Cursor API when `queue_next_submission.py` can chain runs

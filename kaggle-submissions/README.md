# Kaggle Submissions

Archive and workflow for [NeuroGolf 2026](https://kaggle.com/competitions/neurogolf-2026) daily submissions.

## Guides

| Doc | Description |
|---|---|
| [everyday.md](everyday.md) | Per-submission checklist (logs → analysis → plan → implement → submit) |
| [cloud.md](cloud.md) | Run solves remotely (Kaggle Notebook, GitHub Actions, VPS) |

## Layout

```text
kaggle-submissions/
├── README.md
├── everyday.md
├── cloud.md
└── YYYY-MM-DD/
    ├── README.md              # day index + scores
    ├── submission-1/
    │   ├── submission/        # ONNX files
    │   ├── submission_v2.zip
    │   ├── run.log
    │   ├── audit.json
    │   ├── results.json
    │   ├── analysis.md
    │   ├── theory.md
    │   ├── learnings.md
    │   └── kaggle_logs/       # optional
    └── submission-2/
        └── ...
```

## Score history (2026-06-17)

| # | Score | Tasks | Milestone |
|---|---|---|---|
| submission-1 | 835.12 | 64 | M7b slim flip |
| submission-2 | 848.07 | 65 | M8 ARC-GEN routing |
| submission-3 | 848.07 | 65 | M8b autodiscover |
| submission-4 | pending | 70 | M9 ARC-GEN gather fit |

## Quick commands

```bash
# Full submission (local or cloud)
python scripts/run_submission_2026-06-17_s4.py

# Fetch per-task Kaggle logs
python scripts/fetch_kaggle_logs.py \
  --submission_dir kaggle-submissions/2026-06-17/submission-4/submission \
  --out_dir kaggle-submissions/2026-06-17/submission-4/kaggle_logs \
  --download_official

# Audit only
python scripts/audit_submission.py \
  --submission_dir kaggle-submissions/2026-06-17/submission-4/submission \
  --out kaggle-submissions/2026-06-17/submission-4/audit.json \
  --phase 15
```

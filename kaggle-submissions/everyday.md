# Everyday submission workflow

Repeat for each submission **n**.

## Step 1 — Get logs (submission n−1)

```bash
python scripts/fetch_kaggle_logs.py \
  --submission_dir kaggle-submissions/YYYY-MM-DD/submission-{n-1}/submission \
  --out_dir kaggle-submissions/YYYY-MM-DD/submission-{n-1}/kaggle_logs \
  --download_official
```

Store Kaggle leaderboard score in `kaggle_status.txt`.

## Step 2 — Study & document (submission n−1)

In `submission-{n-1}/`:

- `analysis.md` — what changed, audit vs Kaggle, solver mix
- `theory.md` — framework mapping (BLF, CWM, AutoHarness)
- `learnings.md` — distilled takeaways for next submission

## Step 3 — Plan next (submission n)

Create `submission-{n}/`:

- `plan.md` — objective, implementation scope, success criteria
- `strategy.md` — bet, risks, submit rules
- `theory.md` — hypothesis for this run

## Step 4 — Implement & run

```bash
# Local OR cloud — see cloud.md for remote runs
python scripts/run_submission_YYYY-MM-DD_s{n}.py
```

Outputs: `submission/`, `submission_v2.zip`, `run.log`, `audit.json`, `results.json`

## Step 5 — Submit (conditional)

Submit only if:

- `pass_all` increases, OR
- same tasks with higher official score

## Cloud option

Full solves (~90 min) should run remotely. See **[cloud.md](cloud.md)**.

| Local | Remote |
|---|---|
| Code, docs, analysis | `solve_all` + Kaggle submit |
| Pull artifacts after cloud run | Kaggle Notebook / GitHub Actions / VPS |

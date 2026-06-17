# Kaggle Notebook — submission-3 (M8b)

## Run

```bash
python scripts/run_submission_2026-06-17_s3.py
```

Phase 14: autodiscover + seed from submission-2 + cost_audit preserve.

## Pre-scan

```python
from scripts.run_submission_2026-06-17_s3 import prescan_new_tasks
print(prescan_new_tasks())  # expected: []
```

## Autodiscover API

```python
from arc_genome.genome.ops.autodiscover import discover_rules, prescan_new_tasks
from arc_genome.data.arcgen import load_tasks_with_arcgen

tasks = load_tasks_with_arcgen("data/all_tasks.json")
discover_rules(tasks[307]["data"])  # [('bounded_upscale2', 1.0)]
```

## Submit message

```
ARC-Genome M8b: 65 verified, est 999, autodiscover
```

## Fetch Kaggle logs (submission-2 example)

```bash
python scripts/fetch_kaggle_logs.py \
  --submission_dir kaggle-submissions/2026-06-17/submission-2/submission \
  --out_dir kaggle-submissions/2026-06-17/submission-2/kaggle_logs \
  --download_official
```

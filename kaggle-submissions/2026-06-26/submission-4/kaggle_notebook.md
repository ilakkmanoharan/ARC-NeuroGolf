# submission-4 Kaggle notebook runbook

**Phase:** 22 — depth-3 compose + unsolved conv pass  
**Folder:** `kaggle-submissions/2026-06-26/submission-4`

## Environment

```bash
pip install -r requirements.txt
export NEUROGOLF_SKIP_KAGGLE_SUBMIT=1
curl -fsSL https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json
# Restore arc_gen_raw from Dataset or extract from arc_gen.json
```

## Run

```bash
python3 scripts/run_submission_2026-06-26_s4.py
```

## Artifacts

| File | Purpose |
|---|---|
| `submission_v2.zip` | Kaggle upload bundle |
| `results.json` | pass_all, est score, gates |
| `audit.json` | per-task tier + Kaggle est |
| `run.log` | full solve stdout |
| `kaggle_submit_ready.json` | GHA auto-submit trigger (if gates pass) |

## Zip path

`kaggle-submissions/2026-06-26/submission-4/submission_v2.zip`

## Expected runtime

60–75 minutes (326 unsolved tasks, conv second-pass enabled).

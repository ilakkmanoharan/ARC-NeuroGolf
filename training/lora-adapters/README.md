# LoRA adapters (Agent 1)

Three named adapter slots for steps 2–4. **Until fine-tuned**, the Cursor agent uses these names as **personas** + RAG over prior submissions.

| Step | Adapter name | Role | Training folder |
|---|---|---|---|
| 2 | **NeuroGolf-Diagnose** | Analyze logs, hypothesize low score, write theory | `training/lora-diagnose/examples/` |
| 3 | **NeuroGolf-Strategize** | Plan, strategy, hypothesis for next submission | `training/lora-strategize/examples/` |
| 4 | **NeuroGolf-Implement** | Implement solver + run script / notebook | `training/lora-implement/examples/` |

## Export training rows (after each iteration)

```bash
python scripts/export_lora_training_row.py diagnose \
  --submission-dir kaggle-submissions/2026-06-17/submission-4 \
  --input-file kaggle-submissions/2026-06-17/submission-4/kaggle_logs/kaggle_logs.json \
  --input-file kaggle-submissions/2026-06-17/submission-4/results.json \
  --output-file kaggle-submissions/2026-06-17/submission-4/analysis.md

python scripts/export_lora_training_row.py strategize \
  --submission-dir kaggle-submissions/2026-06-17/submission-5 \
  --input-file kaggle-submissions/2026-06-17/submission-4/analysis.md \
  --output-file kaggle-submissions/2026-06-17/submission-5/plan.md

python scripts/export_lora_training_row.py implement \
  --submission-dir kaggle-submissions/2026-06-17/submission-5 \
  --input-file kaggle-submissions/2026-06-17/submission-5/plan.md \
  --output-file scripts/run_submission_2026-06-17_s5.py
```

Fine-tune when you have **20+** examples per adapter (self-hosted inference; not Cursor Cloud today).

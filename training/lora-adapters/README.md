# LoRA adapters (Agent 1)

See design: [private/agent/LoRA-adapters/DESIGN.md](../../private/agent/LoRA-adapters/DESIGN.md) (local)  
Public summary below.

| Step | Adapter | Role |
|---|---|---|
| 2 | **NeuroGolf-Diagnose** | Analyze logs → analysis / theory |
| 3 | **NeuroGolf-Strategize** | Plan next submission (score up/down aware) |
| 4 | **NeuroGolf-Implement** | plan → run_submission script |

## Quick start

```bash
pip install -r requirements-training.txt
python scripts/bootstrap_lora_training_data.py
python scripts/train_lora.py --bootstrap --adapter all
```

Checkpoints: `training/lora-*/checkpoints/`

## ARC-GEN vs LoRA

- **ARC-GEN JSON** → trains ONNX solvers (`arc_genome/`), not the LLM.
- **LoRA** → trains on your `analysis.md`, `plan.md`, run scripts + **Kaggle log summaries**.
- Score **+67** on submission-4 (ARC-GEN-fitted gather) is the positive example Strategize should imitate.

## Export one row manually

```bash
python scripts/export_lora_training_row.py diagnose \
  --submission-dir kaggle-submissions/2026-06-17/submission-4 \
  --input-file kaggle-submissions/2026-06-17/submission-4/kaggle_logs/kaggle_logs.json \
  --output-file kaggle-submissions/2026-06-17/submission-4/analysis.md
```

Fine-tune meaningfully when you have **20+** examples per adapter.

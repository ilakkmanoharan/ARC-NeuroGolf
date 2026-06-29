# LoRA adapter refresh — June 29, 2026

## Problem

LoRA adapters (Diagnose / Strategize / Implement) were trained on older submission patterns (gather, bbox, object programs). They did not yet encode:

- **Prescan gate** — do not solve/submit at 0 hits
- **Dynamic gravity ONNX** — when static gather fails but numpy matches
- **Seed + patch bundle** — skip full solve_all when audit passes
- **Dual submit path** — local `kaggle_auto_submit.py` + GHA fallback

## Fixes

### 1. Bootstrap bug

`scripts/bootstrap_lora_training_data.py` now imports `ADAPTER_GOALS` (was NameError at runtime).

### 2. New training artifacts (submission-3)

| Adapter | Source files |
|---------|--------------|
| Diagnose | `submission-3/analysis.md` |
| Strategize | `submission-3/plan.md`, `strategy.md`, `strategy/June-29-2026/strategy.md` |
| Implement | `scripts/run_submission_2026-06-26_s3.py` |

### 3. Refresh commands

```bash
pip install -r requirements-training.txt

# Export rows from latest submission
python scripts/export_lora_training_row.py diagnose \
  --submission-dir kaggle-submissions/2026-06-26/submission-3 \
  --input-file kaggle-submissions/2026-06-26/submission-3/results.json \
  --input-file strategy/June-29-2026/postmortem-stuck-and-submit.md \
  --output-file kaggle-submissions/2026-06-26/submission-3/analysis.md

python scripts/export_lora_training_row.py strategize \
  --submission-dir kaggle-submissions/2026-06-26/submission-3 \
  --input-file kaggle-submissions/2026-06-26/submission-3/analysis.md \
  --output-file kaggle-submissions/2026-06-26/submission-3/strategy.md

# Rebuild all MLX JSONL + train
python scripts/bootstrap_lora_training_data.py --adapter all
python scripts/train_lora.py --adapter all --iters 200
```

GHA alternative: push changes under `training/lora-*/examples/` → **NeuroGolf train LoRA** workflow on `macos-14`.

### 4. What adapters should learn (score_up pattern)

**Strategize** should propose:

> Phase 21 dynamic gravity when compose finds gravity near-misses; prescan ≥ 1 before solve; seed baseline + N new tasks.

**Diagnose** should flag:

> Static gather compile failure on numpy-matched chains = dynamic rule, not "try depth-3 compose."

**Implement** should emit:

> `arcgen_gravity.py` + `gravity.py` + `kaggle_auto_submit.py` wiring; phase 21 in config.

### 5. Research page

```bash
python scripts/update_lora_research_page.py
```

## Success criteria

- `training/lora-strategize/mlx/train.jsonl` includes Phase 21 / dynamic gravity examples
- `training/lora-*/checkpoints/adapter_config.json` updated after train
- Agent instructions reference **940.75** baseline and Phase 21 prescan gate

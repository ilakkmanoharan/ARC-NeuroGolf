# ARC-NeuroGolf (ARC-Genome)

Minimal neural circuits for ARC-AGI tasks — submission system for the [2026 NeuroGolf Championship](https://kaggle.com/competitions/neurogolf-2026).

## Setup

```bash
pip install -r requirements.txt
mkdir -p data
curl -L https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json
```

## Solve all tasks

```bash
python scripts/solve_all.py --data_file data/all_tasks.json --conv_budget 30
```

## Phased competition runs

Each phase has a theory paper and Kaggle submission under `phases/phaseN/`:

```bash
python scripts/run_phase.py 1   # cost calibration
python scripts/run_phase.py 6   # full pipeline + audit
```

See [phases/README.md](phases/README.md) for results.

## Submit to Kaggle

```bash
python scripts/submit_kaggle.py --zip_path submission.zip --message "ARC-Genome v0.1"
```

## Tests

```bash
python -m pytest tests/ -v
```

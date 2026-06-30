# AGENTS.md

## Cursor Cloud specific instructions

ARC-NeuroGolf (ARC-Genome) is a pure-Python CLI project that builds minimal ONNX
"neural circuits" to solve ARC-AGI tasks for the NeuroGolf Kaggle competition.
There is no web server or frontend — everything runs via `scripts/*.py`.

### Running things

- Use `python3`, not `python` (the README uses `python`, but only `python3`
  exists on this VM).
- Dependencies are refreshed by the startup update script (`pip install -r
  requirements.txt` + `pytest`); no extra install steps are needed.
- Tests: `python3 -m pytest tests/ -v` (fast smoke tests).
- Run the solver end-to-end on a subset (recommended for quick checks):
  `python3 scripts/solve_all.py --data_file data/all_tasks.json --tasks 1,2,3,4,5 --conv_budget 10 --output_dir /tmp/sub --zip_path /tmp/sub.zip`.
  Each task can take several seconds; running all 400 tasks is slow, so scope
  with `--tasks` while developing.

### Data

- `data/all_tasks.json` and `data/arc_gen.json` are committed to the repo and
  already present — do NOT re-download them despite the README's `curl` step.
- `data/arc_gen_raw/` (ARC-GEN-100K raw) is gitignored and absent; only Phase
  6+ / `--arcgen_validation` style flows need it. Standard solving and tests do
  not.

### Kaggle / submission

- `scripts/run_phase.py` and `scripts/submit_kaggle.py` push to Kaggle and need
  `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`). Without those
  secrets, run solve-only flows or pass `--no-submit` to `run_phase.py`.
- There is no configured linter (no ruff/flake8/pyproject config); "lint" for
  this repo effectively means the pytest smoke suite.

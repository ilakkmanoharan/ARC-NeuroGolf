# ARC-NeuroGolf (ARC-Genome)

## Cursor Cloud specific instructions

This is a Python-only, offline batch ML/research project (Kaggle "2026 NeuroGolf
Championship" submission system). There is **no web server, API, database, or
long-running service** — everything runs as one-shot CLI scripts under `scripts/`.

### Running / testing
- Use the system `python3` (3.12). Dependencies install into the user site via
  `pip install -r requirements.txt`. `pytest` is required for tests but is **not**
  in `requirements.txt`; install it separately (the update script handles this).
- Tests: `python -m pytest tests/ -v` (fast, fully offline, no data/secrets needed).
- Standard run commands are documented in `README.md`.

### Non-obvious gotchas
- The solver entrypoint `scripts/solve_all.py` runs all 400 tasks by default and
  is slow (uses `--conv_budget` seconds *per task*). For a quick end-to-end check,
  scope it: `python scripts/solve_all.py --tasks 1,2,3,4,5 --conv_budget 5`.
  It writes ONNX circuits to `--output_dir` and packages them into `--zip_path`.
- `scripts/run_phase.py` and the milestone runners default `--submit` to **True**
  and shell out to the Kaggle CLI. Pass `--no-submit` to stay fully offline; the
  Kaggle path needs `KAGGLE_USERNAME`/`KAGGLE_KEY` (or `~/.kaggle/kaggle.json`).
- Task data files (`data/all_tasks.json`, `data/arc_gen.json`) are already present
  in the repo checkout, so no HuggingFace download is needed despite the README.
- There is no linter configured in this repo (no flake8/ruff/pyproject); CI only
  runs the submission automation, not lint.
- Kaggle credentials (`KAGGLE_USERNAME`/`KAGGLE_KEY`) and `CURSOR_API_KEY` are
  only needed for the submit / CI-automation paths, not for solving or tests.

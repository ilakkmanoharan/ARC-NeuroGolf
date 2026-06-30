# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
ARC-NeuroGolf (a.k.a. ARC-Genome) is an **offline Python CLI/research pipeline** (no web server, database, or GUI). It synthesizes minimal neural circuits as ONNX models for ARC-AGI tasks, validates them with `onnxruntime`, scores them, and packages a `submission.zip` for the Kaggle NeuroGolf 2026 competition.

### Running things
Standard commands are in `README.md`. All scripts are run from the repo root; the `arc_genome` package is imported via `sys.path` insertion, so there is **no install step** for the package itself (no `pyproject.toml`/`setup.py`).

- Tests: `python -m pytest tests/ -v`
- End-to-end solve (core flow): `python scripts/solve_all.py --data_file data/all_tasks.json --conv_budget 30` (add `--tasks 1,2,3` to limit to a quick subset).
- Phased runs: `python scripts/run_phase.py N` (see `phases/README.md`).

### Non-obvious notes
- `pytest` is **not** listed in `requirements.txt` (or `requirements-ci.txt`); it is installed separately by the startup update script. If you re-create the environment manually, `pip install pytest`.
- The task datasets `data/all_tasks.json` and `data/arc_gen.json` are already committed in the repo, so the README's HuggingFace `curl` download is **not needed** in this environment.
- There is **no linter configured** (no flake8/ruff/pylint/black/mypy config); "lint" is not part of the dev workflow here.
- `solve_all.py` reporting some tasks as `UNSOLVED` is expected behavior, not an error — the solver only emits ONNX circuits for tasks it can solve within the cost budget.
- Kaggle submission scripts (`scripts/submit_kaggle.py`, etc.) require `KAGGLE_USERNAME`/`KAGGLE_KEY` (or `KAGGLE_API_TOKEN`) secrets and network access; they are optional and not required for local solve/validate/score.

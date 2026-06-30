# AGENTS.md

## Cursor Cloud specific instructions

This is a single Python (3.12 on the VM) ML/research project: **ARC-NeuroGolf / ARC-Genome**, a batch solver that generates minimal ONNX neural circuits for ARC-AGI tasks and packages them into a `submission.zip`. There is **no server, database, or daemon** — it is a CLI/library run entirely in-process on CPU. "Running the app" means running the solve pipeline.

### Environment
- Use `python3` (there is no `python` alias on this VM). README commands written as `python ...` should be run as `python3 ...`.
- Dependencies (`numpy`, `onnx`, `onnxruntime`, `kaggle`, plus `pytest`) are installed by the startup update script. `onnxruntime` runs CPU-only.
- Data assets are already on disk under `data/` (`data/all_tasks.json` ~1.4 MB, `data/arc_gen.json` ~95 MB), so no HuggingFace download is needed. If `data/all_tasks.json` is ever missing, fetch it per the README.

### Lint / Test / Run
- **Lint:** there is no linter configured in this repo (no ruff/flake8/black config, and CI does not lint).
- **Test:** `python3 -m pytest tests/ -v` (fast smoke tests, builds + validates an ONNX model).
- **Run (end-to-end):** `python3 scripts/solve_all.py --data_file data/all_tasks.json --conv_budget 30`. To run a quick subset, pass `--tasks 1,2,3,4,5` and a small `--conv_budget` (e.g. 10). Output goes to `submission/` and `submission.zip`. A successful run prints `Task NNN: <solver> cost=... score=...` lines and `Created submission.zip: N files`.
- Phase/milestone runners (`scripts/run_phase.py N`, `scripts/run_milestoneN.py`) wrap the same pipeline at different config levels (see `arc_genome/config.py`).

### Caveats
- Solving is CPU-bound and time-boxed by `--conv_budget` (seconds per task), so a full 400-task run is slow; use `--tasks` to scope quick checks.
- Kaggle submission (`scripts/submit_kaggle.py`) and the `scripts/run_submission_*.py` / `submit_*.py` scripts require `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`); they are not needed to solve, validate, score, or test locally.
- The CI-only agent automation (`scripts/trigger_cursor_agent.py`, `requirements-ci.txt` / `cursor-sdk`) needs `CURSOR_API_KEY` and is irrelevant to local development.

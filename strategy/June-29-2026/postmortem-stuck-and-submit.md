# Postmortem — why we got stuck; why auto-submit failed

## Part 1: Why score was stuck (morning → afternoon)

### Symptom

Baseline **940.75** / **72** tasks unchanged through Phases 18–20.

### Timeline of false leads

| Phase | What we tried | Prescan | Why no score |
|-------|---------------|--------:|--------------|
| 18 | Object programs (static gravity + gather) | 0 | Gravity ≠ static gather |
| 19 | Place programs (bbox shift, paste) | 0 | No new compileable rules on unsolved |
| 20 | Depth-2 compose (flip ∘ gravity) | 0 | Numpy match on 32/78; **ONNX compile fails** |

### Technical root cause

**Gravity** compacts colored pixels per column using **runtime** nonzero counts. A single gather index map from example 0 fails when:

- Empty column in ex₀ becomes `const=0`, but ex₁ has pixels there (task **78**)
- Grid extent varies 4×4 / 5×5 / 6×6 (task **32**)

Compose search correctly found rules in numpy; the **compiler** was the bottleneck, not the search.

### Time sinks

- Depth-3 compose prescan (~2.4 min) — confirmed 0
- Full 55-solver prescan — 0
- First gravity ONNX attempt — **4.6M nodes** (full 30×30 unroll) — too large, wrong
- Conv sample on 80 tasks — 0

### Breakthrough

Phase 21: **dynamic column cumsum** + **compile-time `max_grid_extent`** → tasks 32/78 `validate_full=True`.

---

## Part 2: Why submission did not happen automatically

### Expected behavior

Push to `main` including `kaggle-submissions/.../kaggle_submit_ready.json` should run **neurogolf-auto-submit.yml**, which:

1. Copies `submission_v2.zip`
2. Runs `kaggle competitions submit`
3. Sets `results.json` → `"submitted": true`
4. Chains **post-submit** workflow

### What actually happened

| Observation | Implication |
|-------------|-------------|
| GHA **did submit** at `2026-06-29T13:31Z` (`submitted_by: github-actions:auto-submit`) | First push after `kaggle_submit_ready.json` worked |
| Local `results.json` lagged until pull | Dev machine was behind remote |
| Duplicate local CLI submit also ran | Harmless — same zip; Kaggle accepts multiple submits |
| submission-2 was manual CLI | Pattern before GHA secrets were fixed |

### Remaining GHA hardening

| Fix | File |
|-----|------|
| **Local submit helper** | `scripts/kaggle_auto_submit.py` |
| Run script tries **local submit first**, GHA on failure | `scripts/run_submission_2026-06-26_s3.py` |
| **Manual submit completed** | Kaggle CLI — submission-3 uploaded |
| Document dual-path workflow | `strategy/June-29-2026/workflow.md` |

### Verify GHA (operator checklist)

1. GitHub → Settings → Secrets → `KAGGLE_API_TOKEN` or username/key
2. Actions → **NeuroGolf auto-submit** → check failed runs on commit `6f7f385`
3. Re-run workflow manually: `workflow_dispatch` with `submission_dir=kaggle-submissions/2026-06-26/submission-3` (will no-op if already submitted)
4. Future submissions: run script without `NEUROGOLF_SKIP_KAGGLE_SUBMIT` on machine with Kaggle creds **or** fix secrets

---

## Part 3: LoRA gap

Adapters were **not** refreshed with Phase 21 strategy before this session:

- `bootstrap_lora_training_data.py` referenced `ADAPTER_GOALS` without importing it (bug — fixed)
- submission-3 lacked `analysis.md` until today (diagnose bootstrap skipped it)
- Training rows still described Phases 18–19 object/place, not dynamic gravity

See [lora-refresh.md](./lora-refresh.md) for remediation steps.

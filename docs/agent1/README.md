# Agent 1 — Full NeuroGolf loop

Implements [private/agent/agent1.md](../../private/agent/agent1.md): fetch logs → analyze → plan → implement → **GitHub auto-submit** → poll Kaggle every 10 min → repeat.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Cursor Automation: NeuroGolf Agent 1 (steps 1–4)               │
│  LoRA personas: Diagnose → Strategize → Implement               │
└────────────────────────────┬────────────────────────────────────┘
                             │ push kaggle_submit_ready.json + zip
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions: neurogolf-auto-submit.yml (step 5)             │
│  Kaggle submit → commit → chain post-submit                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ poll 10 min (+ retries)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions: neurogolf-post-submit.yml (steps 6–8)          │
│  Fetch logs → commit kaggle_logs.json → push triggers Agent 1     │
└─────────────────────────────────────────────────────────────────┘
```

**Research page (public in repo):**  
https://github.com/ilakkmanoharan/ARC-NeuroGolf/blob/main/kaggle-submissions/research/index.html

**Gallery:**  
https://github.com/ilakkmanoharan/ARC-NeuroGolf/blob/main/kaggle-submissions/gallery/index.html

---

## What you need to do (one-time)

### 1. GitHub secrets

**Settings → Secrets and variables → Actions**

| Secret | Required | Purpose |
|---|---|---|
| `KAGGLE_API_TOKEN` | **Yes** | Auto-submit (step 5) + log fetch |
| `KAGGLE_ARC_GEN_DATASET` | Yes for GHA solve | `ilakkmanoharan/neurogolf-arc-gen` |
| `GMAIL_APP_PASSWORD` | Optional | Email on solve-ready (notify workflow) |

Legacy alternative: `KAGGLE_USERNAME` + `KAGGLE_KEY` instead of `KAGGLE_API_TOKEN`.

### 2. Cursor GitHub integration

- **Cursor Settings → Integrations → GitHub** → authorize `ilakkmanoharan/ARC-NeuroGolf` with **write** access

### 3. Cloud environment (for solve / arc_gen_raw)

- **Cloud Agents → Environments → `ilakkmanoharan/ARC-NeuroGolf`**
- Run **Setup Agent** once so `data/arc_gen_raw` is cached in the environment snapshot

### 4. Create Cursor Automation

1. [cursor.com/automations](https://cursor.com/automations) → **New**
2. **Name:** `NeuroGolf Agent 1`
3. **Trigger:** GitHub → Push to `main`
4. **Instructions:** paste full block from [instructions.md](instructions.md)
5. **Model:** GPT-5 / Composer (high capability)
6. **Tools:** disable **Create pull request**
7. Link **saved environment** `ilakkmanoharan/ARC-NeuroGolf`
8. **Save** → **Enable**

### 5. Enable GitHub Actions workflows

After pushing this repo, these workflows must exist on `main`:

| Workflow | Role |
|---|---|
| `neurogolf-auto-submit.yml` | Step 5 — submit when `kaggle_submit_ready.json` pushed |
| `neurogolf-post-submit.yml` | Steps 6–8 — 10 min poll, fetch logs |
| `notify-kaggle-ready.yml` | Optional issue when manual push-only solve |

---

## Continuous lane (no calendar-day limit)

Submissions run continuously (~1/hour). Kaggle allows 100/day; we target ~24/day.

```text
python3 scripts/submission_lane.py --json   ← active work folder
kaggle-submissions/{active_date}/submission-N/   ← increments N in latest date folder
```

Date folders are archive labels only — **not** tied to calendar today. Post-submit **auto-detects** the latest submitted submission.

---

## LoRA adapter names

| Step | Name | Until trained |
|---|---|---|
| 2 | **NeuroGolf-Diagnose** | Persona + RAG over prior `analysis.md` |
| 3 | **NeuroGolf-Strategize** | Persona + prior `plan.md` style |
| 4 | **NeuroGolf-Implement** | Persona + `run_submission_*` templates |

Export training rows: [training/lora-adapters/README.md](../../training/lora-adapters/README.md)

**Cloud training:** `neurogolf-train-lora.yml` runs on `macos-14` when agent commits rows under `training/lora-*/examples/`. Checkpoints upload as GHA artifacts (gitignored). Agent can also run `train_lora.py` inline on Apple Silicon Cursor cloud.

---

## How to start the loop

### Start or resume the loop

1. **Actions → NeuroGolf post-submit** (auto_detect=true, trigger_cursor_agent=true)
2. Agent solves ~75–90 min → commits `submission_v2.zip` + `kaggle_submit_ready.json`
3. GHA auto-submits to Kaggle
4. Post-submit polls Kaggle **every 10 minutes**, fetches logs, triggers Agent 1 again
5. Repeat continuously

---

## Trigger file (step 5)

Agent commits this to start auto-submit:

```json
// kaggle-submissions/TODAY/submission-N/kaggle_submit_ready.json
{
  "auto_submit": true,
  "message": "ARC-Genome M10a: 71 verified, est 1070, ..."
}
```

Must be pushed together with `submission_v2.zip` and `results.json`.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Auto-submit didn’t run | Check `kaggle_submit_ready.json` in latest push; verify `KAGGLE_API_TOKEN` |
| Agent skips | Read transcript — likely “not today” or already solved |
| Post-submit stuck | Actions → `NeuroGolf post-submit` — may still be polling (10 min intervals) |
| Research page stale | Agent runs `python3 scripts/update_research_index.py` after step 2 |
| LoRA | Not required yet — personas in instructions suffice |

---

## Map to 4-automation setup

You can run **one** Agent 1 automation instead of separate #1–#4, or keep both:

| Agent 1 step | Old automation |
|---|---|
| 1 | #1 fetch logs |
| 2 | #2 analyze (Diagnose LoRA) |
| 3 | #2 plan (Strategize LoRA) |
| 4 | #2 implement + #3 solve |
| 5–8 | GHA auto-submit + post-submit |

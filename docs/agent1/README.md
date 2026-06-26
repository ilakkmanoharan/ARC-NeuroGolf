# Agent 1 — Full NeuroGolf loop

Implements [private/agent/agent1.md](../../private/agent/agent1.md): fetch logs → analyze → plan → implement → **GitHub auto-submit** → wait 1h → repeat.

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
                             │ wait 1h (+ retries)
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
| `neurogolf-post-submit.yml` | Steps 6–8 — 1h wait, fetch logs |
| `notify-kaggle-ready.yml` | Optional issue when manual push-only solve |

---

## Daily folder rule

```text
kaggle-submissions/TODAY/submission-1/   ← first of the day
kaggle-submissions/TODAY/submission-2/   ← same day
```

Agent only works on **today’s UTC date**. Never solves old-date folders.

---

## LoRA adapter names

| Step | Name | Until trained |
|---|---|---|
| 2 | **NeuroGolf-Diagnose** | Persona + RAG over prior `analysis.md` |
| 3 | **NeuroGolf-Strategize** | Persona + prior `plan.md` style |
| 4 | **NeuroGolf-Implement** | Persona + `run_submission_*` templates |

Export training rows: [training/lora-adapters/README.md](../../training/lora-adapters/README.md)

---

## How to start the loop

### First time today

1. **Run Test** on Agent 1 (or wait for post-submit push)
2. Agent bootstraps `TODAY/submission-1` if needed
3. Solve ~75–90 min → commits `submission_v2.zip` + `kaggle_submit_ready.json`
4. GHA auto-submits to Kaggle
5. Post-submit waits **1 hour**, fetches logs, pushes to `main`
6. Push triggers Agent 1 again → step 2

### Manual kick (no logs push)

**Automations → NeuroGolf Agent 1 → Run Test** on `main`

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
| Post-submit stuck | Actions → `NeuroGolf post-submit` — may still be in 1h wait |
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

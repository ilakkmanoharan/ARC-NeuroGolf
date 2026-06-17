# Cloud & Remote Submissions

Run heavy solve + Kaggle submit pipelines **off your Mac** so local RAM and CPU stay free for coding in Cursor.

---

## Why run remotely?

| Resource | Typical full submission |
|---|---|
| Runtime | ~75–90 min CPU |
| Data | ~100 MB (`arc_gen_raw/`) |
| Submission zip | ~2–3 MB |
| Peak memory | ONNX Runtime profiling (heaviest phase) |

Coding, planning, and docs are fine locally. **Full `solve_all` + audit + submit** is what belongs in the cloud.

---

## What stays local vs remote

| Task | Local (Cursor) | Remote (cloud) |
|---|---|---|
| Edit solver code, theory, plans | ✅ | — |
| Full solve (400 tasks) | ❌ | ✅ |
| Kaggle CLI submit | Either | ✅ preferred |
| Fetch logs / write analysis | Either | Either |
| Download artifacts back | ✅ | — |

---

## Option 1: Kaggle Notebook (recommended)

Competition files are already on Kaggle (`task001.json`, `neurogolf_utils.py`, etc.). Best fit for NeuroGolf.

### One-time setup

1. Push `ARC-NeuroGolf` to a **private** GitHub repo.
2. Create a [Kaggle Dataset](https://www.kaggle.com/datasets) from `data/arc_gen_raw/` (~92 MB).
3. Add Kaggle API credentials to the notebook (or use notebook’s built-in competition access).

### Notebook workflow

```bash
# Cell 1 — clone + install
!git clone https://github.com/YOUR_USER/ARC-NeuroGolf.git
%cd ARC-NeuroGolf
!pip install -q -r requirements.txt

# Cell 2 — data
!mkdir -p data
!cp /kaggle/input/neurogolf-arc-gen/arc_gen_raw data/ -r   # your dataset mount
# OR: download all_tasks.json
!curl -L https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json

# Cell 3 — run submission (example: submission-4)
!python scripts/run_submission_2026-06-17_s4.py

# Cell 4 — download artifacts (optional)
from IPython.display import FileLink
FileLink("kaggle-submissions/2026-06-17/submission-4/submission_v2.zip")
```

### Pros / cons

| Pros | Cons |
|---|---|
| Free | Session time limits (~9–12 h) |
| Competition data built-in | Must upload `arc_gen_raw` once as Dataset |
| Submit without touching local machine | Artifacts lost if notebook not saved |

---

## Option 2: GitHub Actions

Push code → workflow runs solve → uploads zip + logs as artifacts → optional Kaggle submit.

### Secrets (repo Settings → Secrets)

| Secret | Value |
|---|---|
| `KAGGLE_USERNAME` | Kaggle username |
| `KAGGLE_KEY` | Kaggle API key |

### Example workflow sketch

```yaml
# .github/workflows/submission.yml
name: NeuroGolf submission
on:
  workflow_dispatch:   # manual “Run” button
    inputs:
      script:
        description: Run script name
        default: run_submission_2026-06-17_s4.py

jobs:
  solve:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Restore arc_gen_raw cache
        uses: actions/cache@v4
        with:
          path: data/arc_gen_raw
          key: arc-gen-raw-v1
      - name: Fetch data if cache miss
        run: |
          if [ ! -d data/arc_gen_raw ]; then
            curl -L ... -o data/ARC-GEN-100K.zip
            unzip data/ARC-GEN-100K.zip -d data/
          fi
      - run: python scripts/${{ inputs.script }}
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
      - uses: actions/upload-artifact@v4
        with:
          name: submission-artifacts
          path: kaggle-submissions/
```

### Pros / cons

| Pros | Cons |
|---|---|
| Fully hands-off after setup | One-time CI wiring |
| Scheduled or manual trigger | Free tier: 6 h job limit (our runs fit) |
| Zero local RAM | Need to cache or fetch `arc_gen_raw` |

---

## Option 3: Cloud VM (VPS)

Small instance (2–4 GB RAM, 2 vCPU) is sufficient.

```bash
ssh user@your-vm
git clone https://github.com/YOUR_USER/ARC-NeuroGolf.git
cd ARC-NeuroGolf
pip install -r requirements.txt
# copy data/arc_gen_raw once (scp or rsync)
python scripts/run_submission_2026-06-17_s4.py
```

**Cost:** ~$5–10/month (Hetzner, DigitalOcean, GCP e2-small).

**Best when:** you want the exact same CLI workflow as local, long runs, no notebook session limits.

---

## Option 4: Cursor Cloud Agents

Use cloud agents for **implementation** (code changes, new solvers). Long 90-minute solves are still better on Kaggle Notebook / VM / Actions.

| Use cloud agent for | Use Kaggle/VM for |
|---|---|
| Writing submission-N code | Running `solve_all` |
| Fixing bugs, docs | Kaggle submit |
| Pre-scan analysis | ONNX profiling load |

---

## Recommended setup

| Phase | Where |
|---|---|
| **Day-to-day coding** | Local Cursor |
| **Full solve + submit** | Kaggle Notebook or GitHub Actions |
| **Artifacts back to repo** | Download zip + `kaggle_logs/` → commit under `kaggle-submissions/YYYY-MM-DD/` |

**Short term:** Kaggle Notebook — fastest to set up, competition-native.  
**Long term:** GitHub Actions with `workflow_dispatch` — push code, click Run, collect artifacts.

---

## One-time checklist

- [ ] Private GitHub repo for `ARC-NeuroGolf`
- [ ] Kaggle Dataset: `data/arc_gen_raw/` (~92 MB)
- [ ] `data/all_tasks.json` (HuggingFace or repo LFS)
- [ ] Kaggle API key in notebook secrets or GitHub Secrets
- [ ] Test run: `python scripts/run_submission_*_sN.py` in cloud
- [ ] Download `kaggle-submissions/<date>/submission-N/` artifacts for local docs

---

## Environment variables

Scripts respect:

```bash
export KAGGLE_BIN=/path/to/kaggle   # default: ~/Library/Python/3.9/bin/kaggle
```

On Linux cloud:

```bash
pip install kaggle
export KAGGLE_BIN=$(which kaggle)
mkdir -p ~/.kaggle
# place kaggle.json with username + key
chmod 600 ~/.kaggle/kaggle.json
```

---

## Daily workflow (with cloud)

See [everyday.md](everyday.md) for the per-submission checklist. When using cloud:

1. Implement submission-N locally in Cursor.
2. Push to GitHub.
3. Trigger cloud run (notebook or Actions).
4. Pull artifacts: `run.log`, `audit.json`, `results.json`, `submission_v2.zip`.
5. Fetch Kaggle logs: `python scripts/fetch_kaggle_logs.py ...`
6. Write `learnings.md`, `theory.md`, plan for submission-(N+1).

Local machine never runs the 90-minute solve.

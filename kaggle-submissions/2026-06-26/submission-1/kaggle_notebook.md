# Kaggle Notebook — submission-1 (M10a)

Reproducible workflow for Phase 16 ARC-GEN-fitted color maps.

---

## Environment

```bash
pip install -r requirements.txt
export NEUROGOLF_SKIP_KAGGLE_SUBMIT=1
mkdir -p data
curl -fsSL https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json
# ARC-GEN samples (required for 100-sample gate):
# unzip data/ARC-GEN-100K.zip -d data/arc_gen_raw
# OR materialize from bundled arc_gen.json:
python3 -c "
import json, os
os.makedirs('data/arc_gen_raw', exist_ok=True)
with open('data/arc_gen.json') as f: arcgen = json.load(f)
for h, ex in arcgen.items():
    with open(f'data/arc_gen_raw/{h}.json','w') as f: json.dump(ex, f)
"
```

---

## Phase 16 solve

```python
from arc_genome.config import set_phase
from arc_genome.solve import solve_all, make_submission_zip

set_phase(16)  # position_gather_arcgen + color_map_arcgen
results = solve_all("data/all_tasks.json", "submission", conv_budget=25)
make_submission_zip("submission", "submission.zip", "data/all_tasks.json")
```

Or run the full pipeline (seed s4 ONNX + pre-scan + solve + audit + conditional submit):

```bash
python3 scripts/run_submission_2026-06-26_s1.py
```

Outputs: `kaggle-submissions/2026-06-26/submission-1/`

---

## Pre-scan (before 90-min solve)

```python
from scripts.run_submission_2026_06_26_s1 import prescan_new_tasks  # or import from run script
# python3 -c "from importlib import import_module; m=import_module('scripts.run_submission_2026-06-26_s1'.replace('-','_')); print(m.prescan_new_tasks())"
```

---

## Audit

```bash
python scripts/audit_submission.py \
  --submission_dir kaggle-submissions/2026-06-26/submission-1/submission \
  --out kaggle-submissions/2026-06-26/submission-1/audit.json \
  --phase 16
```

---

## Kaggle submit

```bash
cp kaggle-submissions/2026-06-26/submission-1/submission_v2.zip submission.zip
kaggle competitions submit -c neurogolf-2026 \
  -f submission.zip \
  -m "ARC-Genome M10a: N verified, est XXX, arcgen color fit"
```

**Submit rule:** `train_only == 0` AND (`pass_all > 70` OR `audit_est >= 1066.52`).

GitHub Actions auto-submits when `kaggle_submit_ready.json` is pushed with `"auto_submit": true`.

---

## Key code paths

| File | M10a change |
|---|---|
| `arc_genome/genome/ops/analytical.py` | `s_color_map_arcgen` |
| `arc_genome/genome/infer.py` | Phase 16 solver ordering |
| `arc_genome/config.py` | Phase 16 `color_map_arcgen` |
| `scripts/run_submission_2026-06-26_s1.py` | Seed s4 + pre-scan + gates |

---

## Submission message format

```
ARC-Genome M10a: {pass_all} verified, est {audit_est}, arcgen color fit
```

Example: `ARC-Genome M10a: 71 verified, est 1075, arcgen color fit`

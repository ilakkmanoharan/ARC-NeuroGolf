# Kaggle Notebook — submission-2 (M10b)

Reproducible workflow for Phase 17 bbox-relative ARC-GEN gather.

---

## Environment

```bash
pip install -r requirements.txt
export NEUROGOLF_SKIP_KAGGLE_SUBMIT=1
mkdir -p data
curl -fsSL https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json
python3 -c "
import json, os
os.makedirs('data/arc_gen_raw', exist_ok=True)
with open('data/arc_gen.json') as f: arcgen = json.load(f)
for h, ex in arcgen.items():
    with open(f'data/arc_gen_raw/{h}.json','w') as f: json.dump(ex, f)
"
```

---

## Phase 17 solve

```python
from arc_genome.config import set_phase
from arc_genome.solve import solve_all, make_submission_zip

set_phase(17)  # bbox_gather_arcgen + position/spatial arcgen + color_map_arcgen
results = solve_all("data/all_tasks.json", "submission", conv_budget=25)
make_submission_zip("submission", "submission.zip", "data/all_tasks.json")
```

Or run the full pipeline (seed submission-1 ONNX + pre-scan + solve + audit + conditional submit):

```bash
python3 scripts/run_submission_2026-06-26_s2.py
```

Outputs: `kaggle-submissions/2026-06-26/submission-2/`

---

## Audit

```bash
python scripts/audit_submission.py \
  --submission_dir kaggle-submissions/2026-06-26/submission-2/submission \
  --out kaggle-submissions/2026-06-26/submission-2/audit.json \
  --phase 17
```

---

## Submit rule

`train_only == 0` AND (`pass_all > 70` OR `audit_est >= 1066.52`).

GitHub Actions auto-submits when `kaggle_submit_ready.json` is pushed with `"auto_submit": true`.

---

## Key code paths

| File | M10b change |
|---|---|
| `arc_genome/genome/ops/arcgen_gather.py` | `s_bbox_gather_arcgen` |
| `arc_genome/onnx/bounded.py` | `compile_bbox_relative_gather` |
| `arc_genome/config.py` | Phase 17 `arcgen_fit_bbox_gather` |
| `scripts/run_submission_2026-06-26_s2.py` | Seed s1 + pre-scan + gates |

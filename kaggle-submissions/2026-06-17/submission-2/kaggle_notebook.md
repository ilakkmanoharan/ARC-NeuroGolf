# Kaggle Notebook — submission-2 (M8)

Reproducible workflow for Phase 13 ARC-GEN routing submission.

---

## Environment

```bash
pip install -r requirements.txt
# Data (if not present):
mkdir -p data
curl -L https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json
# ARC-GEN samples (required for 100-sample gate):
unzip data/ARC-GEN-100K.zip -d data/arc_gen_raw
```

---

## Phase 13 solve

```python
from arc_genome.config import set_phase
from arc_genome.solve import solve_all, make_submission_zip

set_phase(13)  # bounded_world + slim flip + arcgen_routing
results = solve_all("data/all_tasks.json", "submission", conv_budget=25)
make_submission_zip("submission", "submission.zip", "data/all_tasks.json")
```

Or run the full pipeline (solve + audit + conditional submit):

```bash
python scripts/run_submission_2026-06-17_s2.py
```

Outputs: `kaggle-submissions/2026-06-17/submission-2/`

---

## Pre-scan (numpy, before 90-min solve)

```python
from scripts.run_submission_2026-06-17_s2 import prescan_new_tasks
print(prescan_new_tasks())  # e.g. [307] for bounded_upscale2
```

---

## Audit

```bash
python scripts/audit_submission.py \
  --submission_dir kaggle-submissions/2026-06-17/submission-2/submission \
  --out kaggle-submissions/2026-06-17/submission-2/audit.json \
  --phase 13
```

---

## Kaggle submit

```bash
cp kaggle-submissions/2026-06-17/submission-2/submission_v2.zip submission.zip
~/Library/Python/3.9/bin/kaggle competitions submit -c neurogolf-2026 \
  -f submission.zip \
  -m "ARC-Genome M8: N verified, est XXX, arcgen routing"
```

**Submit rule:** pass_all > 64 OR (pass_all = 64 AND score > 835.12).

---

## Key code paths

| File | M8 change |
|---|---|
| `arc_genome/data/arcgen_meta.py` | ARC-GEN signature detection |
| `arc_genome/genome/belief.py` | `rank_solver_order()` |
| `arc_genome/genome/ops/bounded.py` | `s_bounded_upscale2` |
| `arc_genome/onnx/bounded.py` | `compile_dynamic_upscale2()` |
| `arc_genome/config.py` | Phase 13 `arcgen_routing` |

---

## Submission message format

```
ARC-Genome M8: {pass_all} verified, est {audit_est}, arcgen routing
```

Example: `ARC-Genome M8: 65 verified, est 998, arcgen routing`

---

## Leaderboard check

```bash
~/Library/Python/3.9/bin/kaggle competitions submissions list -c neurogolf-2026 --csv
```

Saved to `kaggle_status.txt` after automated submit.

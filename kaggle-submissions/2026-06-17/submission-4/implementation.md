# Implementation — submission-4

## Code changes

### `arc_genome/genome/ops/arcgen_gather.py` (new)

- `_position_idx_arcgen(td)` — fit idx on train+test+arc-gen
- `_spatial_idx_arcgen(td)` — fit idx+cst with constants
- `s_position_gather_arcgen(td)` → `build_gather_model`
- `ARCgen_GATHER_SOLVERS` registered in harness

### `arc_genome/config.py`

```python
if level >= 15:
    cfg.arcgen_fit_gather = True
```

### `arc_genome/genome/infer.py`

```python
if cfg.arcgen_fit_gather:
    solvers = ARCgen_GATHER_SOLVERS + solvers
```

### `scripts/run_submission_2026-06-17_s4.py`

- `set_phase(15)`
- Seed from submission-3
- Pre-scan for 5 candidates
- Submit if pass_all > 65

## Run

```bash
python scripts/run_submission_2026-06-17_s4.py
```

## Pre-scan validation

```python
# Expected output:
# Pre-scan new pass_all: 5 [83, 108, 211, 326, 385]
```

## Tests

```bash
python3 -c "
from arc_genome.config import set_phase
from arc_genome.genome.ops.arcgen_gather import s_position_gather_arcgen
from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
from arc_genome.onnx.model import save_model
import tempfile
set_phase(15)
tasks = load_tasks_with_arcgen('data/all_tasks.json')
for tn in [83,108,211,326,385]:
    m = s_position_gather_arcgen(tasks[tn]['data'])
    assert m is not None
"
```

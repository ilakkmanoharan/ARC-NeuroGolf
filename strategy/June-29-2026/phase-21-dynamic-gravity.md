# Phase 21 — Dynamic gravity ONNX

## Problem

Milestone 5 / Phase 18 `_s_gravity_dir` builds a **static** gather index from example 0:

- Empty column in ex₀ → `cst[r,c] = 0`
- Same column filled in ex₁ → gather reads 0 instead of input pixel

Numpy gravity is **correct**; static gather is **wrong**.

## Algorithm (numpy reference)

```python
in_grid = max_channels(input) > 0      # includes background cells
colored = sum_channels_1_to_9(input) > 0
gh, gw = extent(in_grid)               # top-left grid size

# gravity_up, column c:
k = count(colored[0:gh, c])
for r_out in range(gh):
    if r_out >= k: output[r_out,c] = background
    else: output[r_out,c] = input[s,c] where s is (r_out+1)-th colored row in column
```

`gravity_down` places the same `k` pixels in rows `gh-k .. gh-1`.

## ONNX implementation

**File:** `arc_genome/onnx/gravity.py`

| Component | Ops |
|-----------|-----|
| `colored` | `Slice` ch1–9, `ReduceSum` |
| `gh, gw` | `ReduceMax`, `_bbox_scalars` (from bounded.py) |
| Column cumsum | Per-column chained `Add` on cell scalars |
| Rank match | `Equal(cumsum[s], rank_target)`, `Greater` boundary masks |
| Output | Per-channel weighted sum of `match[s] * input[ch,s,c]` |
| Background ch0 | Fill when slot empty (`r_out >= k` up, `r_out < gh-k` down) |
| Outside extent | Zero (matches `to_onehot` padding) |

### Compile-time bounding

`max_grid_extent(task_data)` = max train+test height/width. Loops run `0..max_h-1` only; positions beyond baked extent output zero.

| Task | max extent | Nodes (approx) |
|------|------------|----------------|
| 78 | 10×10 | ~165k |
| 32 | 6×6 | ~52k |

### Bugs fixed during bring-up

1. Batch slice `[0,ch,r,c]` → `[1,ch+1,r+1,c+1]` (was zero batch)
2. Up boundary: `k > r_out` not `k-1 > r_out`
3. Down boundary: `r_out >= lb` via `r_out > lb-1`; `r_out < gh` via `gh - r_out > 0`

## Solver wiring

**File:** `arc_genome/genome/ops/arcgen_gravity.py`

```python
ARCgen_GRAVITY_SOLVERS = [
    ("gravity_up_dynamic", s_gravity_up_dynamic),
    ("gravity_down_dynamic", s_gravity_down_dynamic),
]
```

- Config: `arcgen_dynamic_gravity` @ level **21**
- `infer.py`: gravity solvers run **before** compose (cheap rule match, targeted compile)

## Validation

```bash
set_phase(21)
validate_full(task32, gravity_down_dynamic)  # True
validate_full(task78, gravity_up_dynamic)      # True
```

Prescan on 328 unsolved: **tasks 32, 78** only (other gravity tasks need different rules or chains).

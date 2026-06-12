# Phase 1: Calibrated Cost Metrology

## Abstract

Phase 1 establishes **trustworthy local scoring** for the ARC-Genome compiler. Without accurate cost estimation, optimization is blind: our v0.1.1 submission scored 346.95 on Kaggle while local estimates predicted ~2,700 — a 8× discrepancy proving the cost model was wrong.

## Theory

NeuroGolf scoring is MDL-style:

```text
score(task) = max(1, 25 − ln(cost))
cost        = |θ| + mem(bytes) + MACs
```

The competition treats ONNX as a **physical circuit** whose price is the sum of three currencies:

1. **Parameters** — weight storage in initializers
2. **Memory** — footprint of tensors the graph materializes
3. **MACs** — multiply-accumulate operations during execution

### What was special about Phase 1

Phase 1 is not about solving more tasks. It is **metrology** — building instruments that measure circuit cost the same way Kaggle does.

Key concepts introduced:

- **Shape propagation** — infer tensor dimensions through Conv, Slice, Pad, ArgMax, OneHot chains
- **MAC accounting per op-type** — Conv scales as `C_out × H × W × C_in × K²`; elementwise ops scale with tensor volume
- **Cost-gated selection** — reject candidates below score threshold 8.0 when cheaper alternatives exist
- **Probe models** — identity, 1×1 conv, 3×3 conv baselines in `probes/` for Kaggle calibration

### Hypothesis

Large-kernel conv pipelines score ~1 on Kaggle because official MAC counting traverses full `[1,10,30,30]` tensors through ArgMax → OneHot → Pad. Our old estimator undercounted MACs by ~100×.

### Implementation

| Module | Change |
|---|---|
| `arc_genome/onnx/cost.py` | Shape-aware MAC + memory model |
| `arc_genome/config.py` | Phase 1 flags: `calibrated_cost`, `cost_gate_min_score=8` |
| `arc_genome/genome/infer.py` | Multi-candidate selection by minimum cost |
| `scripts/build_probes.py` | Calibration probe ONNX generators |

### Expected outcome

Phase 1 may not dramatically raise Kaggle score alone, but enables Phases 2–6 to optimize the correct objective.

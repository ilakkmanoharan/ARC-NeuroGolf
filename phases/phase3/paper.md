# Phase 3: Analytical Genome Expansion

## Abstract

Phase 3 grows the **operator genome** from 11 to 18+ primitives plus parameterized family solvers. The thesis: top leaderboard teams solve ~200 tasks analytically at near-zero cost; we had 6.

## Theory

### The genome abstraction

Each ARC task implements an operator φ. The genome is a minimal program:

```text
Genome = (operator_sequence, parameters)
ONNX   = compile(Genome)
```

Phase 3 expands the basis set of compile-targets.

### New primitives

| Operator | Semantics | ONNX emission |
|---|---|---|
| `translate` | Shift content by (dx, dy) | Gather index map |
| `scale_down` | Subsample by integer factor | Gather |
| `pad_embed` | Embed small grid in larger canvas | Slice + Pad |
| `mask_preserve` | Keep input where nonzero, fill elsewhere | Mul + Add mask |
| `mirror_complete` | Reflect half to complete symmetry | Gather |

### Family solvers

Parameterized templates search small configuration spaces:

- `color_then_translate` — recolor then spatial shift
- `extract_recolor` — nonzero mask → uniform color

Families are **not** per-task code. They are search over `(dx, dy, color_map)` etc.

### ARC-GEN alignment (conceptual)

google/ARC-GEN maps 400 tasks to ~50 procedural generators. Phase 3 lays groundwork: family solvers approximate generator-level reasoning without cloning the full ARC-GEN repo.

### What was special

Phase 3 shifts the compiler from **reactive** (try known solvers) to **generative** (search parameterized transformation families). Coverage should rise while keeping cost low.

### Expected impact

+30–80 new analytical wins, replacing expensive conv fallbacks. Target Kaggle score > 4,000.

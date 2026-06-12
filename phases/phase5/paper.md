# Phase 5: Hard Task Recovery — Extended Search

## Abstract

Phase 5 targets the **107 unsolved tasks** from v0.1.1 with a second-pass pipeline: extended time budgets, larger kernel cap (11) only for failures, and conv_v2 multi-strategy fallback.

## Theory

### Two-speed compilation

```text
Pass 1 (fast):  analytical + conv(kernel ≤ 7), 25s budget
Pass 2 (slow):  conv_v2(kernel ≤ 11), 90s budget, unsolved only
```

This mirrors branch-and-bound: cheap methods first, expensive search only where necessary.

### Conv v2

Sequential attempt of:
1. `conv_fixed` with extended kernel
2. `conv_variable` with extended kernel  
3. `conv_diffshape` with extended kernel

Still capped at kernel 11 — never returns to 29×29 madness.

### Task bucketing (conceptual)

| Bucket | Strategy |
|---|---|
| Kernel too small (phase 2) | Extended kernel in pass 2 |
| Compositional | Phase 4 composition (re-run) |
| Object reasoning | Family solvers |
| Truly hard | Conv v2 + extended budget |

### What was special

Phase 5 accepts that **coverage and cost trade off** — pass 2 tolerates higher cost only when pass 1 failed entirely (score 0). Solved tasks are not re-convolved with large kernels.

### Expected impact

+30–60 newly solved tasks. Total coverage 330–370. Kaggle score > 5,500 combined with cheaper Phase 2–4 wins.

# Phase 2: Structural Compilation — Killing Expensive Conv

## Abstract

Phase 2 attacks the dominant failure mode of v0.1.1: **287 of 293 wins were bloated conv nets** with kernels up to 29×29. We reframe the compiler to prefer structural ONNX graphs and cap conv search.

## Theory

### The conv trap

Least-squares conv fitting is a universal approximator on ARC grids — but universality is expensive. A 29×29 conv on 30×30×10 costs ~75M MACs per layer. The competition rewards **recognition + compilation**, not **fitting**.

### Concepts introduced

1. **Kernel budget** — search only sizes {1, 3, 5, 7, 9, 11}; reject 13+ kernels entirely
2. **Content normalization** — crop examples to tight bounding boxes before analytical matching; many "hard" tasks are simple transforms on embedded content
3. **Weight sparsification** — post-fit pruning: zero |w| < ε, shrink kernel to minimal bounding box of nonzeros
4. **Conditional Pad** — skip Pad node when output already fills 30×30 (saves MACs + memory)
5. **Prefer structural** — Gather/Slice/Transpose score orders of magnitude below conv

### Compilation principle

```text
φ : Grid → Grid
```

If φ is composable from structural ops, emit those. Conv is a **last resort**, not a default.

### What was special

Phase 2 is the first phase that directly targets **Kaggle score per solved task**, not coverage. Even at constant 293 coverage, cutting median cost from 10⁷ to 10³ could raise total score from ~350 to ~2,000+.

### Implementation

| Flag | Effect |
|---|---|
| `max_kernel=7` | Cap conv search |
| `content_normalize=True` | Bbox crop before solvers |
| `conv_sparsify=True` | Prune + shrink conv weights |
| `prefer_structural=True` | Cost-ordered candidate pick |

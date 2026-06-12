# Phase 4: Compositional Program Search

## Abstract

Phase 4 introduces **depth-limited composition search** over a numpy primitive library. Many ARC tasks are not atomic transforms — they are chains like `rot90 → tile2x2 → color_map`.

## Theory

### Composition as search

```text
Find sequence (op₁, op₂, …, opₖ) such that
  opₖ ∘ … ∘ op₁(input) = output   ∀ training pairs
```

Search space: `|P|^d` where |P| ≈ 9 primitives, d ≤ 3 → ~729 chains — tractable per task.

### Primitive basis

| Primitive | Grid function |
|---|---|
| identity | g |
| flip_h | flipud(g) |
| flip_v | fliplr(g) |
| rot90 / rot180 | np.rot90 |
| transpose | g.T |
| tile2x2 | np.tile(g, (2,2)) |
| upscale2 | nearest-neighbor ×2 |
| crop_center | center crop |

### Pruning strategy

- Skip chains starting with identity (depth > 1)
- Verify all train + test pairs simultaneously
- On match, emit ONNX via `spatial_gather` (pixel remapping)

### Compilation path

```text
numpy chain verified → spatial_gather(td) → minimal Gather ONNX
```

We do not compose ONNX nodes directly (hard to minimize cost). We **verify in program space, emit in circuit space**.

### What was special

Phase 4 bridges symbolic AI and neural compilation: the intelligence is program search; the artifact is still a tiny ONNX graph.

### Expected impact

+20–40 tasks that no single primitive handles. Raises analytical fraction above 50%.

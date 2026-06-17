# Theory — 2026-06-14 Submission (Phase 11 / M6)

## Framework: Bounded Code World Models

M6 applied **Code World Models** (Murphy) to ARC compilation:

```text
Observations (train I/O)
    → infer world-model code (bounded_flip)
    → verify on train + test + ARC-GEN
    → compile to dynamic ONNX
    → submit if pass_all
```

### Why static Gather failed (M3–M5 plateau)

Phases 1–10 used **absolute Gather indices**. Variable-shape flips map `(r,c)` to different source cells per grid size → single static idx cannot represent the rule.

### M6 breakthrough: runtime bbox

```text
gh = last row with occupancy + 1
gw = last col with occupancy + 1
flip within [0:gh, 0:gw] → embed in 30×30
```

ONNX computes gh/gw via unrolled ReduceMax + weighted Max (no banned NonZero/Loop).

### BLF (Bayesian belief update)

| Evidence | Effect |
|---|---|
| Train+test match flip rule | High prior on bounded_flip |
| 100 ARC-GEN passes | Posterior confirms → accept |
| Gravity on task 32 (M7 later) | ARC-GEN fails → reject |

### ASRA (action semantics)

Flip semantics refined with **bbox context**:
- Old: `fliplr(grid)` on absolute coordinates
- New: `fliplr(grid[0:gh, 0:gw])` with runtime gh, gw

### Cost pathology

The unrolled dynamic compiler trades **expressiveness for circuit size**:
- 900 cells × per-cell Sub/Cast/Gather chain
- 10 channel-wise Gathers (redundant — fixed in M7b slim)

Score formula `max(1, 25 − ln(params + memory))` penalizes memory-heavy graphs even when param count is low (1,590 params but 251k memory).

---

## Lesson

**Representation unlocks coverage; cost optimization unlocks score.**

M6 added tasks (+2). Next step: same 64 tasks, cheaper circuits → higher total without new coverage.

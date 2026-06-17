# Theory — submission-1 (Phase 12 / M7b)

**Date:** 2026-06-17  
**Kaggle:** 835.12 (64 tasks)  
**Change:** Slim bounded flip compiler

---

## Problem

M6 bounded flip (tasks 150/155) was **correct but expensive**:

```text
score = max(1, 25 − ln(params + memory))
```

| | M6 flip | Impact |
|---|---|---|
| params | 1,590 | Low |
| memory | 251,220 | **Dominates cost** |
| Kaggle score | 12.56 | Worst in submission |

Root cause: unrolled 30×30 dynamic index map + **10 redundant per-channel Gather nodes**.

---

## Solution: shared spatial Gather (AutoHarness)

Flip is a **spatial** transform — all 10 color channels permute identically.

```text
Before (M6):
  for k in 0..9:
    out[k] = Gather(flat(input[k]), flat_idx)   # 10 Gathers

After (M7b):
  flat = Reshape(input, [1, 10, 900])
  out  = Gather(flat, flat_idx, axis=2)       # 1 Gather
  out  = Mul(out, bbox_mask)
```

**AutoHarness principle:** don't repeat orchestration when semantics are shared across channels.

---

## Code World Model unchanged

The world model (`bounded_flip` within runtime bbox) is identical. Only the **compilation target** changed:

| Layer | M6 | M7b |
|---|---|---|
| Semantics | `fliplr(grid[0:gh,0:gw])` | same |
| gh/gw detection | ReduceMax unroll | same |
| Index map | 900 dynamic cells | same |
| Channel gather | 10× redundant | **1× shared** |

---

## BLF (belief update)

| Belief | M6 | M7b |
|---|---|---|
| P(correct \| ARC-GEN) | 1.0 | 1.0 |
| E[Kaggle score \| circuit] | 12.56/flip | **12.90/flip** |

Correctness posterior unchanged. Value posterior updated → select slim variant.

---

## ASRA

Action semantics unchanged. **Compilation context** refined: spatial actions apply uniformly across color channels in onehot encoding.

---

## Verified outcome

| Metric | Predicted | Actual |
|---|---|---|
| Flip task score | 12.90 | 12.90 (audit) |
| Total Kaggle | ~835.1 | **835.12** |
| New tasks | 0 | 0 |

Theory validated: cost-only optimization, monotonic score improvement.

---

## Remaining cost pathology

Slim flip still has ~11,168 nodes (gh/gw unroll dominates). Further wins:

1. **Shared bbox subgraph** — compute gh/gw once, reference everywhere
2. **M8 ARC-GEN routing** — new coverage, not cost tuning
3. **Object programs** — multi-step CWM for remaining 336 unsolved

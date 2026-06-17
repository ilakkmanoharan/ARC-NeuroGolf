# Theory — submission-2 (Phase 13 / M8)

**Date:** 2026-06-17  
**Change:** ARC-GEN signature routing + bounded dynamic upscale2

---

## Problem

submission-1 proved **cost tuning alone cannot break the 64-task ceiling** (+0.68 Kaggle). Coverage requires new representations that generalize across variable ARC-GEN grids.

Key blocker: `s_upscale` requires **fixed input shape** across train+test. Task 307 has variable shapes (3×3, 2×2, 4×4, 5×5) but consistent **2× nearest-neighbor upscale** — passes 100 ARC-GEN samples in numpy but was unreachable by static gather.

---

## Solution 1: ARC-GEN signature routing (BLF)

```text
arc-gen examples → detect primitive signatures (upscale2, flip, color_map, …)
                 → prior over solver families
                 → reorder harness enumeration (belief-ranked)
```

| Component | Role |
|---|---|
| `arcgen_meta.py` | Signature detection from arc-gen samples |
| `belief.rank_solver_order` | Map signatures → solver priority |
| `infer._solver_list(td)` | Belief-ordered enumeration when Phase 13 |

**AutoHarness principle:** try hypotheses that ARC-GEN evidence supports *first*, not blind alphabetical order.

---

## Solution 2: Bounded dynamic upscale2 (CWM)

Same recipe as bounded flip (M6/M7b):

```text
1. Runtime gh/gw from occupancy ReduceMax unroll
2. Output mask: r < 2×gh, c < 2×gw
3. Gather index: flat_idx = (r//2)*GW + (c//2)
4. Single shared-channel Gather (slim, like Phase 12 flip)
```

| Layer | Static upscale | bounded_upscale2 |
|---|---|---|
| Input shape | fixed | **variable** |
| Scale | any sh×sw | 2× (ARC-GEN signature) |
| ARC-GEN gate | N/A | 100-sample verify |

Numpy reference: `_dynamic_upscale2_oh` in `bounded.py`.  
ONNX compiler: `compile_dynamic_upscale2()` in `onnx/bounded.py`.

---

## BLF update

| Belief | Prior (arc-gen) | Evidence (train) | Posterior |
|---|---|---|---|
| upscale2 | signature match > 0.5 | train+test match | compile bounded_upscale2 |
| flip_h/v | signature match | train match | compile bounded_flip |

ARC-GEN routing does **not** lower the 100-sample gate — it directs search toward hypotheses likely to pass it.

---

## ASRA

Generator-conditioned semantics: "upscale" on variable-shape tasks means **scale content bbox by 2×**, not tile a fixed 30×30 canvas. Routing discovers this context from arc-gen primitive fits.

---

## Verified outcome

| Metric | Predicted | Actual |
|---|---|---|
| New task | 307 | 307 |
| Kaggle Δ | +12.94 | **+12.95** |
| Total Kaggle | ~848 | **848.07** |

Theory validated: bounded upscale2 + ARC-GEN routing. Audit ratio 84.9% stable.

---

## Phase config

```python
set_phase(13)  # Phase 12 + arcgen_routing=True
```

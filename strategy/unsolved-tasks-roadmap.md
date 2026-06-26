# Unsolved tasks (~330) and the path beyond conv search

At **70 pass_all** tasks, roughly **330** competition tasks still lack a circuit that passes **train + test + ARC-GEN[:100]**. They are not one problem — they are many transformation types the current symbolic vocabulary does not yet express as cheap, generalizing ONNX graphs.

This document records the strategy for cracking that pool and where **LoRA adapters** fit.

---

## Why conv search stops working

`arc_genome/genome/infer.py` tries solvers in deliberate order: bounded → extended → family → milestone5 → **ARC-GEN gather/color** → analytical → conv fallbacks.

Convolution solvers (`conv_diff`, `conv_fixed`, `conv_var`) are the **expensive last resort**. They can fit train/test locally but often **fail ARC-GEN** — inflating local “solved” counts without earning Kaggle points.

**Key lesson (submission-4):** Phase 15 **ARC-GEN-fitted position gather** delivered **+5 pass_all** and **+67 Kaggle** (848 → 915). The audit delta tracked Kaggle almost exactly. Symbolic + ARC-GEN gating beats more conv search.

**Key lesson (submission-1):** Phase 16 color_map_arcgen preserved 70 tasks but **0 prescan hits** — color-only extensions are exhausted at the current ceiling.

---

## How to crack the unsolved pool

### Proven loop (repeat per family)

```text
1. Observe failure mode in kaggle_logs / audit (train_ok but not pass_all, prescan miss, unsolved)
2. Hypothesize a compact program family (gather, color, object, compose)
3. Fit parameters over train + test + arc-gen[:100]
4. Compile to minimal static ONNX
5. Prescan new solver on UNSOLVED tasks only (before 90-min solve_all)
6. Seed baseline ONNX (preserve existing pass_all tasks)
7. solve_all with new solver prioritized before conv
8. Submit only if pass_all rises and train_only == 0
```

### Near-term solver families

| Phase | Family | What it captures | Code area |
|-------|--------|------------------|-----------|
| **17** (active) | Bbox-relative ARC-GEN gather | Gather indices break when content bbox shifts | `arcgen_gather.py`, `bounded.py`, `infer.py` |
| **18–19** | Object extract → transform → place | Crop bbox, recolor, gravity, paste back | `milestone5.py` primitives — need composed programs |
| **20+** | Multi-step composition | 2–3 op chains with ARC-GEN gate | `compose` + depth search |
| **Ongoing** | Conv → analytical swap | Replace expensive conv with cheaper symbolic circuits | Per-task cost audit |

### Failure buckets in the unsolved pool

| Bucket | Symptom | Solver direction |
|--------|---------|------------------|
| Layout-relative copy | train gather works, ARC-GEN fails | Bbox-relative gather (Phase 17) |
| Object motion | gravity / slide / align | `s_gravity_*_var` + dynamic bbox |
| Object selection | largest component, hollow rect | `s_largest_component`, `s_hollow_rect` |
| Multi-object | extract A, transform, place on B | New object-program compiler |
| Structural | tile, compose, conditional | Depth-2/3 composition search |

Each family is typically **50–200 lines** of fitter + ONNX compiler — same pattern as `arcgen_gather.py`, not a new end-to-end ML model.

### Prescan discipline

Before every `solve_all`, run the candidate solver on unsolved tasks only:

```text
prescan s_bbox_gather_arcgen → expect 3–8 hits if hypothesis is right
if prescan empty → pivot family; do not burn 90 min on conv
if prescan hits   → solve_all with seeded baseline (keep 70 pass_all)
```

Logged in `results.json` as `prescan_new_candidates`.

---

## Realistic score roadmap

```text
70 pass_all  ──Phase 17 bbox gather──►  73–78   (low risk, incremental)
              ──Object programs──────►  85–95   (extract/place + ARC-GEN)
              ──Composition depth 3────►  100+    (multi-step Code World Models)
              ──Conv → analytical swap─►  higher avg score per task
```

All 400 tasks is a multi-milestone compiler project. **80–100 pass_all** is achievable with 3–5 more prescan-validated symbolic families if each repeats the submission-4 pattern.

---

## Can LoRA adapters help?

**Yes — indirectly.** They do **not** solve grids or compile ONNX.

### What LoRA trains on

```text
Kaggle grade → logs → Agent artifacts (analysis.md, plan.md, run scripts)
    → export_lora_training_row.py → training/lora-*/examples/
    → bootstrap_lora_training_data.py → MLX JSONL
    → train_lora.py (Llama-3.2-3B via MLX)
```

Synthetic JSONL rows come **from** submission research, not from training LoRA. ARC-GEN JSON grids train ONNX solvers in `arc_genome/` separately.

### Three adapters, one goal (raise Kaggle score)

| Adapter | Step | Helps the 330 by… |
|---------|------|-------------------|
| **NeuroGolf-Diagnose** | After logs | Clustering failures: train_ok/not pass_all, unsolved, conv-without-generalization |
| **NeuroGolf-Strategize** | Before plan | Proposing the **next symbolic family** with expected pass_all delta; imitating `score_up` runs (s4) vs avoiding `score_flat` (s1) |
| **NeuroGolf-Implement** | Before solve | Scaffolding `run_submission_*.py`, phase flags, prescan wiring |

### What LoRA cannot do

- Fit gather indices or emit ONNX (that is `arc_genome/`)
- Replace ARC-GEN validation (competition gate)
- Generalize from tens of examples to 330 diverse ARC tasks without growing the scored dataset

### When LoRA becomes material

| Dataset size | Effect |
|--------------|--------|
| ~16 rows today (per adapter) | Structures agent loop; personas + score_goal prompts |
| 20+ rows per adapter | Fine-tuned Strategize starts beating hand-written plans |
| Every scored submission | `score_up` / `score_flat` labels teach which families to repeat vs abandon |

---

## Decision summary

| Question | Answer |
|----------|--------|
| How to solve ~330 tasks? | Add **symbolic solver families** (gather variants → object programs → composition), each ARC-GEN-gated and prescan-tested |
| More conv search? | **No** — high cost, low Kaggle yield at 70 pass_all |
| Can LoRA help? | **Yes** as research memory and plan selection; **no** as a grid solver |
| What moves Kaggle? | `arc_genome/` compiler work; LoRA accelerates **which** family to build next |

---

## References

- submission-4 analysis (gather +67 Kaggle): `kaggle-submissions/2026-06-17/submission-4/analysis.md`
- submission-2 plan (Phase 17): `kaggle-submissions/2026-06-26/submission-2/plan.md`
- Infer orchestrator: `arc_genome/genome/infer.py`
- LoRA design (private copy): `private/solver-strategy/` (local) · public LoRA summary: `training/lora-adapters/README.md`

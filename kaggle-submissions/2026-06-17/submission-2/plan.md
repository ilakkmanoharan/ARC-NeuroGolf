# Submission Plan — submission-2 (M8)

**Baseline:** submission-1 — 64 tasks, **835.12** Kaggle  
**Date:** 2026-06-17

---

## Objective

Break the **64-task coverage ceiling** — submission-1 proved cost tuning (+0.68) is insufficient for meaningful progress. submission-2 targets **new pass_all tasks** via ARC-GEN-aware solver routing.

---

## Primary: ARC-GEN hex routing (Phase 13)

### Hypothesis

Each task hex in `data/arc_gen.json` maps to a generator family. Routing the harness to **family-specific bounded world models first** unlocks tasks where generic search misses the rule.

```text
task hex → arc_gen.json metadata → prior over {flip, gravity, color, conv, …}
         → try bounded CWM family first → ARC-GEN verify → compile
```

### Implementation scope

| Component | File | Purpose |
|---|---|---|
| Generator index | `arc_genome/data/arcgen_meta.py` | Parse arc_gen.json, cluster by transform signature |
| BLF router | `arc_genome/genome/belief.py` | Score hypotheses per task from train + generator prior |
| Harness wire | `arc_genome/genome/infer.py` | Route bounded solvers by belief rank |
| Phase 13 config | `arc_genome/config.py` | `arcgen_routing=True` |

### Target tasks (from submission-1 learnings)

| Pattern | Status in M7 | M8 goal |
|---|---|---|
| bounded flip | 150, 155 solved | find more flip variants |
| bounded gravity | fails ARC-GEN | route only when generator matches |
| train_only position_gather | 12 tasks | generator-conditioned idx may pass ARC-GEN |
| variable-shape unsolved | 336 tasks | scan + route |

### Success criteria

| Metric | submission-1 | submission-2 target |
|---|---|---|
| pass_all tasks | 64 | **≥ 68** |
| Kaggle score | 835.12 | **≥ 870** |
| New solver wins | 0 | **≥ 4** |

---

## Secondary: shared bbox subgraph (if time permits)

submission-1 slim gather saved 29% memory on flip. Remaining bottleneck: **gh/gw unroll** (~11k nodes).

Dedupe bbox computation into a single shared subgraph referenced by all 900 index cells.

| | submission-1 flip | target |
|---|---|---|
| memory | 179,220 | < 100,000 |
| score (tasks 150/155) | 12.90 | **≥ 14** |

Only implement if M8 routing doesn't fill the run budget.

---

## Out of scope

| Item | Reason (from submission-1 learnings) |
|---|---|
| Third-pass conv | 0 wins on 336 unsolved across M4–M6 |
| M5 Gather primitives | 0 pass_all |
| Gravity without routing | task 32 train-only overfit |
| Lower ARC-GEN gate | M1 lesson |

---

## Run

```bash
python scripts/run_submission_2026-06-17_s2.py   # to be created
```

Outputs to `kaggle-submissions/2026-06-17/submission-2/`.

**Submit only if:** pass_all > 64 OR (pass_all = 64 AND Kaggle > 835.12).

---

## Estimated runtime

~90 min full solve (same as submission-1). Pre-scan routing candidates in numpy before full compile to avoid wasted ONNX builds.

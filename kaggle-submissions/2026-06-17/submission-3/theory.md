# Theory — submission-3 (Phase 14 / M8b)

**Date:** 2026-06-17  
**Baseline:** 848.07 Kaggle, 65 tasks  
**Change:** Autodiscover bounded rule harness

---

## Problem

submission-2 found task 307 via **manual** arc-gen scan. 335 tasks remain unsolved. Manual scan does not scale; submission-2 autodiscover post-analysis found **0 unsolved rule hits** but exposed rules on solved tasks (rot180, transpose, rot90).

We need a **systematic discovery loop**:

```text
rule library → numpy verify (train+test+100 ARC-GEN) → compile → ONNX → score
```

---

## Solution: Autodiscover registry

`arc_genome/genome/ops/autodiscover.py`:

```text
BOUNDED_RULES: [(name, fn_onehot)]
discover_rules(td) → [(solver_name, 1.0)]
prescan_new_tasks(solved, tasks) → [task_nums]
```

Rules encode **bbox-relative** transforms on onehot tensors — same semantics as bounded CWM compilers.

### Harness integration (Phase 14)

When `level >= 14`, `_solver_list(td)` ranks solvers by autodiscover confidence (+0.2 boost) before generic arcgen_meta priors.

### Seed baseline

Copy submission-2 ONNX before solve; `cost_audit_task` ensures monotonic official score selection. Prevents bounded_rot180 (~12.9) from displacing compose (~13.39) or transpose (25.0).

---

## Kaggle score theory (validated)

```text
score = max(1, 25 − ln(memory + params))
Kaggle_total ≈ 0.849 × audit_est
```

submission-2: est 998.5 → actual 848.07. Delta +12.95 = one task at 12.94. **Theory exact.**

---

## Why 0 new tasks expected

Autodiscover library covers single-step bbox transforms. Unsolved 335 tasks require:

- Multi-object reasoning (extract, select, compose)
- Color logic with spatial context
- Rules that fail onehot encoding (mask_preserve class)

Extending the rule library without object programs won't unlock them.

---

## Phase config

```python
set_phase(14)  # phase 13 + autodiscover routing + third_pass disabled
```

---

## Expected outcome

| Metric | Value |
|---|---|
| pass_all | 65 |
| Kaggle | ~848 |
| Infrastructure | autodiscover wired for submission-4 |

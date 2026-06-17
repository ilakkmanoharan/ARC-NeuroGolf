# Submission Plan — submission-3 (M8b)

**Baseline:** submission-2 — 65 tasks, **848.07** Kaggle  
**Date:** 2026-06-17

---

## Objective

Deploy **systematic autodiscover** (Phase 14) and validate no regression while preparing bounded-library expansion. Score target: maintain 65 pass_all; stretch +2–5 via cheaper bounded graphs if compile improves.

---

## Primary: Autodiscover harness (Phase 14)

### What

```text
BOUNDED_RULES (numpy) → 100 ARC-GEN verify → solver priority in harness
                      → prescan before 90-min solve
```

| Component | File |
|---|---|
| Rule registry | `arc_genome/genome/ops/autodiscover.py` |
| Harness wire | `arc_genome/genome/infer.py` (phase ≥ 14) |
| Config | `arc_genome/config.py` phase 14 |

### Rules in library

| Rule | Solver | Known tasks |
|---|---|---|
| flip_h/v | bounded_flip | 150, 155 |
| upscale2 | bounded_upscale2 | 307 |
| rot90/180 | bounded_rot* | 380, 87, 140 (solved, cheaper alt exists) |
| transpose | bounded_transpose | 179, 241 (transpose analytical wins) |
| scale_down2 | bounded_scale_down2 | 0 tasks |

### Pre-scan result (before run)

**0 new tasks** on 335 unsolved. Autodiscover is infrastructure for submission-4+.

---

## Secondary: Seed + cost audit

Copy submission-2 ONNX as baseline; `cost_audit_task` only replaces if official score improves. Prevents regression on transpose (25.0) and compose (13.39) tasks.

---

## Out of scope

| Item | Reason |
|---|---|
| Third-pass conv | Disabled in phase 14 — 0 wins |
| mask_preserve fix | Raw-grid rule ≠ onehot semantics |
| Bounded rot compile | More expensive than compose; no new tasks |

---

## Success criteria

| Metric | submission-2 | submission-3 target |
|---|---|---|
| pass_all | 65 | **≥ 65** |
| Kaggle | 848.07 | **≥ 848** |
| New tasks | 1 (307) | **0 expected** |
| Autodiscover prescan | — | wired + logged |

---

## Run

```bash
python scripts/run_submission_2026-06-17_s3.py
```

Submit if pass_all ≥ 65 and audit est ≥ 998.4 (no regression).

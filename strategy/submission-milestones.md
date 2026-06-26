# Submission milestones — what each phase proved

Score-targeted history for strategy decisions. Full artifacts under `kaggle-submissions/`.

| Submission | Phase | pass_all | Kaggle | Outcome | Lesson |
|------------|-------|---------:|-------:|---------|--------|
| 2026-06-17 s1–s2 | early | 64–65 | 835–848 | score_up | Baseline symbolic stack + seeding |
| 2026-06-17 s3 | 14 | 65 | 848.07 | score_flat | Autodiscover infra; no new coverage |
| 2026-06-17 **s4** | **15** | **70** | **915.03** | **score_up** | **ARC-GEN-fitted gather: +5 tasks, +67 Kaggle** |
| 2026-06-26 s1 | 16 | 70 | 915.03 | score_flat | color_map_arcgen safe but 0 prescan new |
| 2026-06-26 **s2** | **17** | *pending* | *pending* | *in progress* | Bbox-relative ARC-GEN gather bet |

## Patterns to imitate (`score_up`)

- Fit symbolic parameters over **train + test + arc-gen[:100]**, not train/test alone.
- Prescan before solve; only ship when `prescan_new_candidates` non-empty or audit est rises with `train_only == 0`.
- Seed prior submission ONNX; never regress pass_all tasks.

## Patterns to avoid (`score_flat`)

- Color-only or conv-only extensions when prescan returns empty.
- Submitting when local est moves but pass_all count is unchanged vs baseline.

## Next milestones (planned)

| Milestone | Target | Mechanism |
|-----------|--------|-----------|
| M10b (s2) | 73–78 pass_all | Phase 17 bbox-relative gather |
| M11 | 80+ pass_all | Object extract/place programs |
| M12 | 100+ pass_all | Multi-step composition (CWM) |

See [unsolved-tasks-roadmap.md](./unsolved-tasks-roadmap.md) for full strategy.

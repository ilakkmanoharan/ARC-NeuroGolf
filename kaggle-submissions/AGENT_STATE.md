# Agent state — read at step 0 (Imi-style PM)

Last updated: 2026-06-29 — submission-4 blocked flat; submission-3 submitted.

## Baseline (latest scored)

| Field | Value |
|-------|------:|
| Folder | `kaggle-submissions/2026-06-17/submission-4` (north-star) |
| Kaggle public | **915.03** |
| pass_all | **70** |

## Latest submitted (Kaggle pending)

| Field | Value |
|-------|------:|
| Folder | `kaggle-submissions/2026-06-26/submission-3` |
| pass_all | **74** |
| est | **1115** |
| New tasks | **32**, **78** |
| Status | submitted 2026-06-29 — await GHA log fetch |

## Last solve (blocked)

| Field | Value |
|-------|-------|
| Folder | `kaggle-submissions/2026-06-26/submission-4` |
| Phase | **22** depth-3 compose + unsolved conv pass |
| Status | **blocked** — 74 pass_all flat vs s3, no `kaggle_submit_ready.json` |
| Runtime | ~5.5 h (326 unsolved × 2 passes), 0 new tasks |

## Submit gates

```text
BLOCKED s4: pass_all=74 == s3, est=1115.0 < 1116.0
Conv second-pass on 326 unsolved: 0 ARC-GEN-safe wins
Primary submit path: GHA neurogolf-auto-submit on kaggle_submit_ready.json
```

## Do not repeat

- Phase 16 color_map_arcgen alone (0 prescan at 70)
- Phase 17 bbox gather (already shipped in s2)
- Submit flat — north star is Kaggle score up
- Full conv pass on unsolved when symbolic prescan empty (s4 proved 0 gain)

## Next lever

1. Post-submit fetch when Kaggle grades submission-3 → PATH LOGS diagnose
2. If s3 Kaggle > 915: pivot conv→analytical swap or new symbolic family
3. If s3 flat on Kaggle: revisit composition compiler / object programs

## References

- `private/strategy-from-internet/aimar-haddadi-mapping.md`
- `strategy/unsolved-tasks-roadmap.md`
- `strategy/June-29-2026/`

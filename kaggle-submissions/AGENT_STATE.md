# Agent state — read at step 0 (Imi-style PM)

Last updated: 2026-06-29 — submission-3 submitted (local CLI).

## Baseline (latest scored)

| Field | Value |
|-------|------:|
| Folder | `kaggle-submissions/2026-06-26/submission-2` |
| Kaggle public | **940.75** |
| pass_all | **72** |

## Active work

| Field | Value |
|-------|-------|
| Latest submit | `kaggle-submissions/2026-06-26/submission-3` |
| Phase | **21** dynamic gravity |
| Status | **submitted** 2026-06-29 — 74 pass_all, est 1115 |
| New tasks | **32**, **78** |

## Submit automation

```text
Primary: scripts/kaggle_auto_submit.py (local Kaggle CLI)
Fallback: GHA neurogolf-auto-submit on kaggle_submit_ready.json push
Verify repo secrets: KAGGLE_API_TOKEN
```

## Next

Post-submit fetch when Kaggle grades submission-3. See `strategy/June-29-2026/`.

## Do not repeat

- Phase 16 color_map_arcgen alone (0 prescan at 70)
- Phase 17 bbox gather (already shipped in s2)
- Submit flat — north star is Kaggle score up

## References

- `private/strategy-from-internet/aimar-haddadi-mapping.md`
- `strategy/unsolved-tasks-roadmap.md`

# June 29, 2026 — NeuroGolf session (complete record)

**North star:** Kaggle public score up (baseline **940.75** → target **~954** with submission-3).

| Doc | Contents |
|-----|----------|
| [strategy.md](./strategy.md) | Decision tree, phases, anti-patterns |
| [workflow.md](./workflow.md) | End-to-end agent + GHA + LoRA loop |
| [phase-21-dynamic-gravity.md](./phase-21-dynamic-gravity.md) | Compiler technical spec |
| [postmortem-stuck-and-submit.md](./postmortem-stuck-and-submit.md) | Why stuck; why auto-submit failed |
| [prescan-results.md](./prescan-results.md) | Audit numbers |
| [lora-refresh.md](./lora-refresh.md) | Adapter training with today's strategies |
| [session-log.md](./session-log.md) | Chronological log |

## Outcome (updated 2026-06-29 evening)

| Item | Status |
|------|--------|
| Kaggle submit | **Done** — local CLI (74 pass_all, est 1115) |
| GHA auto-submit | Failed — see [postmortem-stuck-and-submit.md](./postmortem-stuck-and-submit.md) |
| LoRA refresh | Bootstrap + train run; Phase 21 rows added |
| Next | Post-submit fetch logs when Kaggle grades |

## Code map

```
arc_genome/onnx/gravity.py           # dynamic gravity compiler
arc_genome/genome/ops/arcgen_gravity.py
arc_genome/genome/ops/arcgen_compose.py   # Phase 20 (0 new hits)
arc_genome/genome/ops/arcgen_object.py    # Phase 18
arc_genome/genome/ops/arcgen_place.py     # Phase 19
scripts/kaggle_auto_submit.py        # local + fallback submit
scripts/run_submission_2026-06-26_s3.py
strategy/June-29-2026/
```

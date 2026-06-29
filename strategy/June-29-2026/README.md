# June 29, 2026 — NeuroGolf session (complete record)

**North star:** Kaggle public score up (baseline **940.75** → target **2000+** via coverage).

| Doc | Contents |
|-----|----------|
| [score-flat-diagnosis.md](./score-flat-diagnosis.md) | **Why 940.75 unchanged** — 1.44 MB ONNX cap |
| [roadmap-2000.md](./roadmap-2000.md) | Path to 2000–3000 (conv scale, rogermt) |
| [strategy.md](./strategy.md) | Decision tree, phases, anti-patterns |
| [workflow.md](./workflow.md) | End-to-end agent + GHA + LoRA loop |
| [phase-21-dynamic-gravity.md](./phase-21-dynamic-gravity.md) | Compiler technical spec |
| [postmortem-stuck-and-submit.md](./postmortem-stuck-and-submit.md) | Phase 18–21 journey; GHA submit |
| [prescan-results.md](./prescan-results.md) | Audit numbers |
| [lora-refresh.md](./lora-refresh.md) | Adapter training with today's strategies |
| [session-log.md](./session-log.md) | Chronological log |

## Outcome (updated 2026-06-29 evening)

| Item | Status |
|------|--------|
| submission-3 Kaggle | **940.75** — unchanged (tasks 32/78 **oversized**) |
| Root cause | Phase 21 gravity ONNX: 2.5 MB + 8.5 MB &gt; **1.44 MB** limit |
| Fix | Audit size gate + submission-4 conv pass on unsolved |
| Next | GHA **submission-4** (Phase 22, `run_submission_2026-06-26_s4.py`) |

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

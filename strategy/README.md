# NeuroGolf solver strategy

Public research record for how ARC-Genome breaks the **~330 unsolved task** ceiling and how **LoRA adapters** support the loop.

| Document | Contents |
|----------|----------|
| [unsolved-tasks-roadmap.md](./unsolved-tasks-roadmap.md) | Why conv search stops working, solver-family roadmap, prescan discipline, LoRA role |
| [submission-milestones.md](./submission-milestones.md) | Scored submission history and what each phase proved |

**North star:** increase Kaggle public score on [neurogolf-2026](https://www.kaggle.com/competitions/neurogolf-2026) via **symbolic solver families + ARC-GEN validation**, not brute-force conv fitting.

**Current baseline (June 2026):** 915.03 Kaggle, 70 pass_all — [2026-06-17 submission-4](../kaggle-submissions/2026-06-17/submission-4/).

**Active bet:** Phase 17 bbox-relative ARC-GEN gather — [2026-06-26 submission-2 plan](../kaggle-submissions/2026-06-26/submission-2/plan.md).

Related: [Solver strategy](../strategy/README.md) · [LoRA adapters](../training/lora-adapters/README.md) · [Agent 1 loop](../docs/agent1/README.md) · [LoRA research charts](../kaggle-submissions/research/lora/index.html)

# June 29, 2026 — NeuroGolf strategy session

**North star:** increase Kaggle public score (baseline **940.75** / **72** pass_all from submission-2).

| Doc | Purpose |
|-----|---------|
| [strategy.md](./strategy.md) | Why Phase 21, decision tree, submit gates |
| [phase-21-dynamic-gravity.md](./phase-21-dynamic-gravity.md) | Technical design + compile algorithm |
| [session-log.md](./session-log.md) | Chronological work log (Phases 18–21) |
| [prescan-results.md](./prescan-results.md) | Latest prescan numbers and next steps |

## Outcome (this session)

Phase 21 **dynamic gravity ONNX** breaks the Phase 18–20 compile wall:

| Task | Rule | Solver | ARC-GEN |
|------|------|--------|---------|
| **32** | `gravity_down` | `gravity_down_dynamic` | pass |
| **78** | `gravity_up` | `gravity_up_dynamic` | pass |

**Prescan:** **+2** new unsolved hits → submission-3 may proceed to `solve_all` (cost audit still required).

## Key insight

Static gather cannot express gravity when per-column nonzero **count varies across examples** (empty column in ex₀ ≠ empty in ex₁). Dynamic ONNX uses runtime column cumsum + rank matching within bbox extent `gh×gw`.

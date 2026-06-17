# Strategy — 2026-06-17

## Submission discipline

| Rule | Rationale |
|---|---|
| Submit only when score or task count improves | M3–M5 wasted 4 slots at identical 809.32 |
| Keep 100 ARC-GEN gate | M1 lesson: 199 solve → 23 earn |
| Official score selection | Pick circuits Kaggle actually rewards |
| One submission per real change | 100/day limit is not the bottleneck |

## Today's bet

**Score-per-task optimization** on bounded flip — proven +0.34/task in local official scorer.

Not betting on:
- Gravity (ARC-GEN failure confirmed)
- Third-pass conv (0 wins on 336 unsolved in M4–M6)
- New Gather primitives (M5 saturated)

## Path to 1,000+

| Phase | Lever | Est. impact |
|---|---|---|
| M7b ✅ today | Slim flip | +0.7 score |
| M8 | ARC-GEN hex routing | +10–30 tasks |
| M8b | Shared bbox subgraph | +5–10 avg on bounded |
| M9 | Object programs | +20–50 tasks |

## Kaggle limits today

- **maxDailySubmissions:** 100
- **Used recently:** 7 on 2026-06-14
- **Today's submission:** 1 (this run)

## Success criteria

| Metric | Pass |
|---|---|
| pass_all | ≥ 64 |
| Kaggle score | > 834.44 |
| Flip tasks 150/155 | pass_all with score ≥ 12.9 |

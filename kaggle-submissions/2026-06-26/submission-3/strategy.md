# Submission strategy — submission-3 (Phase 21)

## North star

Increase Kaggle public score vs **940.75** / **72** pass_all.

## Winning bet: dynamic gravity ONNX

Phases 18–20 treated gravity as static gather. Phase 21 compiles **runtime column cumsum** within dynamic grid extent — the correct abstraction for tasks where per-column nonzero count varies across ARC-GEN examples.

## Methods used today

| Method | Role |
|--------|------|
| ARC-GEN prescan | Gate before `solve_all` — reject 0-hit phases |
| `max_grid_extent` | Cap ONNX graph size per task |
| Seed + patch | Copy submission-2 (72) + add 2 new ONNX — skip 90min full solve |
| Bundle cost audit | Official Kaggle scorer on 74-task zip |
| `kaggle_auto_submit.py` | Local submit fallback when GHA does not run |

## Anti-patterns (do not repeat)

- Submit when prescan = 0 on unsolved
- Static gather for gravity / compose near-misses
- Full 400-task `solve_all` when bundle audit already passes with seeds

## Next submission hypothesis

Depth-2 **flip ∘ dynamic_gravity** for task 32 alternate chain; prescan before solve.

# Theory — submission-3 (Phase 18 / M11)

## Hypothesis

~328 tasks remain unsolved after submission-2. Many failure buckets are **object-centric**:

- gravity / slide (non-zero pixels fall or shift)
- largest connected component isolation
- hollow or filled bounding rectangles per color
- keep-one-color / remove-one-color masks

Milestone 5 already had these rules on train+test only. Phase 18 refits gather compilation on **train + test + ARC-GEN[:100]** so solvers generalize like `bbox_gather_arcgen`.

## Expected lift

Moderate (+3–8 pass_all) if object rules explain a slice of the pool. Zero prescan is acceptable — confirms need for composed programs (M12).

## Why now

submission-2 proved incremental symbolic families beat conv search. Bbox gather exhausted its prescan niche; object programs are the documented next family in `strategy/unsolved-tasks-roadmap.md`.

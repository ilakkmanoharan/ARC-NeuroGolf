# Theory — submission-5 (M13 compact slim gravity)

## Mechanism

Kaggle counts a task only when the ONNX file is **≤ 1.44 MB** and passes official validation. submission-3 added tasks 32/78 with dynamic gravity graphs that exceeded the limit — local `pass_all` rose to 74 but **effective** task count stayed 72.

submission-5 uses `build_gravity_model_slim`:

1. **Single-Gather spatial index** — one gather per output cell instead of multi-channel unroll.
2. **Background ch0 fill** — occupancy from channels 1–9 only; background within bbox.
3. **Compile-time bounding** — `max_h × max_w` from train+test caps graph size (32→6×6, 78→10×10).

This preserves the dynamic bbox semantics from Phase 21 while shrinking node count ~3–10×.

## Why Kaggle score should rise

| Component | submission-2 | submission-5 (expected) |
|---|---:|---:|
| Counted tasks | 72 | **74** |
| Tasks 32+78 score (est) | 0 (rejected) | ~12.4 + ~12.3 |
| Audit total est | 1091 | **1116** |
| Public (× ~86% ratio) | 940.75 | **~965** |

The audit/Kaggle ratio on submission-2 was **940.75 / 1091 ≈ 86.2%**. Applying the same ratio to est 1116 → **~962–968**.

## Size gate enforcement

- `infer._try_record` rejects oversized when `use_official_score`
- `write_kaggle_submit_ready.py` blocks submit if any pass_all ONNX exceeds limit
- Run script seeds only size-safe ONNX from submission-2 zip

## What this does not solve

326 tasks remain unsolved. Gravity prescan is **saturated** (only 32/78). Symbolic prescan (object, place, compose-2/3) returned **0** on unsolved in s4. Further pass_all gains require new compiler families or cost swaps on existing conv circuits.

# Strategy — submission-3 (M11)

**Goal:** Object extract/transform/place with ARC-GEN gate  
**Bet:** gravity + component + rect rules generalize where bbox gather stopped

---

## Strategic framing

| Horizon | Target | Mechanism |
|---|---|---|
| submission-2 (done) | 72 pass_all | bbox-relative ARC-GEN gather |
| **submission-3** | **75–80 pass_all** | ARC-GEN object programs |
| M12 | 100+ pass_all | multi-step Code World Models |

---

## AutoHarness v9

```text
1. set_phase(18)
2. seed submission-2 ONNX (72-task baseline)
3. prescan ARCgen_OBJECT_SOLVERS on unsolved tasks
4. solve_all — object arcgen solvers before conv_diff
5. package validate_full pass_all ONNX only
6. audit phase 18
7. submit if pass_all > 72 OR est >= 1092.2, train_only == 0
```

---

## Guardrails

| Risk | Mitigation |
|---|---|
| Rule overfit train+test only | Fit indices on `_arcgen_examples` (100 samples) |
| Regression vs s2 | Seed 72 baseline ONNX; cost-audit |
| Zero prescan hits | Document pivot to compose depth-2 |
| GHA submit fail | `NEUROGOLF_SKIP_KAGGLE_SUBMIT=1` + manual fallback |

---

## Aimar mapping (discipline, not Mamba)

- **World model** = ARC-GEN prescan before 90-min solve  
- **Synthetic memory** = LoRA export after post-submit  
- **Agent PM** = `kaggle-submissions/AGENT_STATE.md`

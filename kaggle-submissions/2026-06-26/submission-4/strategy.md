# Strategy — submission-4 (M12)

**Phase:** 22  
**Seed:** `kaggle-submissions/2026-06-26/submission-3/submission_v2.zip`  
**Date folder:** 2026-06-26

---

## Execution order

```text
0. set_phase(22) — depth-3 compose + dynamic gravity (4 dirs)
1. Extract arc_gen_raw from arc_gen.json if missing
2. Seed 74 baseline ONNX into submission-4/submission/
3. Prescan: compose_arcgen + ARCgen_GRAVITY + object + place on unsolved
4. solve_all(data, OUT, conv_budget=25, task_nums=unsolved_326)
5. make_submission_zip → submission_v2.zip
6. audit_submission.py --phase 22
7. Gate: pass_all > 74 AND train_only == 0 AND est >= 1116
8. write kaggle_submit_ready.json if PASS else notes.md BLOCKED
```

---

## Environment

```bash
export NEUROGOLF_SKIP_KAGGLE_SUBMIT=1
pip install -r requirements.txt
curl -fsSL https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json
# arc_gen_raw from data/arc_gen.json if needed
```

---

## Gates

| Check | Threshold |
|---|---|
| pass_all | > 74 |
| train_only | == 0 |
| kaggle_score_est | >= 1116.0 |
| vs north-star 915.03 | implied by pass_all increase |

---

## Runtime budget

- Prescan: ~5 min
- solve_all 326 unsolved: ~60–75 min (conv second-pass enabled)
- Audit: ~2 min

**Do not stop early.**

---

## Submit message template

`ARC-Genome M12: {N} verified, est {est}, unsolved conv pass`

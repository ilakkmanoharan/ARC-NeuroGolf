# Roadmap to Kaggle score 2000–3000

**Current:** 940.75 public (72 effective tasks)  
**Target:** 2000–3000  
**Date:** 2026-06-29

---

## Score math

Per task: `score = max(1, 25 - ln(cost))` where  
`cost = params + memory_bytes + MACs`.

| Metric | Value |
|--------|------:|
| Current total | 940.75 |
| Effective tasks | ~72 |
| Avg per task | ~13.1 |
| Perfect task (cost→0) | 25 |
| Floor (huge cost) | 1 |

**To reach 2000** at current average (~13.1/task):

```text
2000 / 13.1 ≈ 153 passing tasks  (+81 from today)
```

**To reach 2000** at rogermt average (~14.9/task):

```text
2000 / 14.9 ≈ 134 passing tasks  (+62 from today)
```

**To reach 3000:**

```text
3000 / 14.9 ≈ 201 tasks  (+129 from today)
```

Coverage dominates. Compiler tweaks on 72 tasks cannot reach 2000 unless every task scores ~28 (impossible — max is 25).

---

## What top solvers do (literature + leaderboard)

### rogermt — 276/400, ~4106 points ([HF repo](https://huggingface.co/rogermt/ARC-AGI))

Solver order (cheap → expensive):

1. Identity, color_map, transpose/flip/rotate  
2. Tiling, upscale, concat patterns  
3. Position LUT, spatial_gather  
4. **One-hot convolution** — **220 tasks** (bulk of coverage)

Key: conv encodes colors as 10-channel one-hot, learns kernel via least squares, decodes with ArgMax. Kernel sizes tried smallest-first.

### ashhhhhh26 enhanced solver

- Analytical solvers first (near-zero cost)  
- `conv_budget` 25–60 on unsolved  
- ~127–293/400 depending on time budget  

### Competition constraints (non-negotiable)

- Static shapes only  
- Banned ops: Loop, Scan, NonZero, Unique, Script, Function  
- **Max 1.44 MB per ONNX file** ← submission-3 failure mode  
- Validation: train + test + ARC-GEN-100K + private slice  

### ARC-AGI program synthesis (papers)

- **DreamCoder** / library learning — compositional programs  
- **AlphaCode / MCTS** — search over program space  
- **NeuroGolf-specific:** bounded compose depth + ARC-GEN gate is our synthesis loop; conv is the neural fallback  

Our Phase 18–22 symbolic stack matches rogermt tiers 1–3. We are missing **conv at scale** on unsolved (submission-3 skipped full `solve_all`).

---

## Staged plan

### Stage A — Fix measurement (submission-3 postmortem)

- [x] Diagnose flat score → **1.44 MB limit**  
- [ ] Audit gate: `file_size_ok` + `kaggle_eligible` tier  
- [ ] Never submit oversized ONNX  

**Expected Kaggle delta:** 0 (process fix)

### Stage B — submission-4 (M12, Phase 22)

```text
Seed 72 valid ONNX (submission-2 baseline, not oversized s3 extras)
Prescan depth-3 compose + 4-way dynamic gravity on 326 unsolved
solve_all(unsolved, conv_budget=25)   # ~60–90 min on GHA
Gate: pass_all > 72 AND all files ≤ 1.44 MB AND train_only = 0
```

**Conservative:** +1–5 tasks → **~955–1010** Kaggle  
**Optimistic:** +8–15 tasks → **~1050–1150** Kaggle  

Automate: `gh workflow run neurogolf-submission.yml` with `run_submission_2026-06-26_s4.py`.

### Stage C — Conv scale-up (submission-5+)

| Knob | Current | Target |
|------|---------|--------|
| `conv_budget` | 25 | 60–90 |
| `second_pass` | 90s | 180s |
| `unsolved_max_kernel` | 9 | 15–29 |
| Tasks attempted | prescan-gated | **all 328 unsolved every run** |

Reference: rogermt solves 220 via conv alone. Even 50 new conv tasks ≈ **+650** Kaggle at ~13/task.

**Target after Stage C:** 120–150 tasks → **~1600–2000**

### Stage D — Analytical expansion

- Hidden gather (train+test+ARC-GEN fit): tasks 83, 108, 211, 326, 385 (~+57 est, verify size)  
- Compressed dynamic gravity ONNX (&lt;1.44 MB) for 32, 78  
- Depth-4 compose with pruned search  
- Cost audit: swap high-cost conv → analytical where pass preserved  

**Target:** 180–220 tasks → **2500–3000** (rogermt territory)

### Stage E — LoRA agent loop

Adapters must internalize:

1. **Size gate** before est celebration  
2. **Conv second-pass** when symbolic prescan = 0  
3. **Coverage math** — 2000 needs ~80+ new tasks, not +2 phantom  

See [lora-refresh.md](./lora-refresh.md).

---

## Anti-patterns

| Pattern | Why it fails |
|---------|--------------|
| Submit at est 1115 with 2 oversized files | Kaggle stays 940.75 |
| Symbolic prescan only, skip conv | 0 new tasks for weeks |
| Depth-N compose without size check | Correct numpy, reject ONNX |
| Optimize cost on 72 tasks only | Max +200 Kaggle even if perfect |

---

## Success metrics per submission

| Gate | Threshold |
|------|-----------|
| `kaggle_eligible` count | &gt; prior Kaggle-effective count |
| All ONNX | ≤ 1.44 MB |
| `train_only` | 0 |
| Public score (post-grade) | &gt; 940.75 |

Do not select submission for final leaderboard until public score confirms gain.

---

## Automation (close laptop)

```bash
# 1. Push docs + code (audit size gate, s4 script)
git push origin main

# 2. Run submission-4 on GHA (~90 min)
gh workflow run neurogolf-submission.yml \
  -f date=2026-06-26 \
  -f submission_number=4 \
  -f run_script=run_submission_2026-06-26_s4.py \
  -f fetch_prev_logs=true \
  -f auto_commit=true

# 3. If gates pass → kaggle_submit_ready.json → neurogolf-auto-submit
# 4. post-submit polls logs → agent PR for submission-5
```

Optional LoRA train: push `training/lora-*/examples/` → **neurogolf-train-lora.yml** on macos-14.

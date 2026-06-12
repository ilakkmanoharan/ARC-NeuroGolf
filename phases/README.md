# ARC-Genome Competition Phases

Six incremental submissions to [NeuroGolf 2026](https://kaggle.com/competitions/neurogolf-2026), each with a theory paper and Kaggle submit via CLI.

## Run a phase

```bash
python scripts/run_phase.py <1-6> --conv_budget 20
```

## Results summary

| Phase | Focus | Solved | Zip size | Kaggle submit |
|---|---|---|---|---|
| 1 | Calibrated cost metrology | 294/400 | 3.5 MB | ✓ |
| 2 | Kernel cap (≤11) + sparsify | 199/400 | 1.1 MB | ✓ |
| 3 | Extended analytical genome | 199/400 | 1.1 MB | ✓ |
| 4 | Composition search | 199/400 | 1.1 MB | ✓ |
| 5 | Second-pass conv v2 | 199/400 | 1.1 MB | ✓ |
| 6 | Cost audit + ARC-GEN gate | 199/400 | 1.1 MB | ✓ |

**Trade-off:** Phase 1 maximizes coverage (294 tasks). Phases 2–6 sacrifice ~95 tasks for much cheaper ONNX (67% smaller submission). Check Kaggle submissions page for which phase scores higher.

## Folder structure

```text
phases/
├── phase1/   paper.md, probes/, submission.zip, results.json
├── phase2/   paper.md, ...
...
└── phase6/   paper.md, ...
```

## Baseline comparison

- v0.1.1 (pre-phases): 293 solved, Kaggle score **346.95**
- Phase 1: best coverage (294), calibrated cost model

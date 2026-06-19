# Theory — submission-1 (Phase 16 / M10a)
Bootstrapped from kaggle-submissions/2026-06-17/submission-5 on 2026-06-19.

**Change:** ARC-GEN-fitted color-map parameters

---

## Hypothesis

Some same-shape tasks are not solved by train/test color-map inference because the small examples do not expose every relevant input color. ARC-GEN samples can reveal the missing color pairs before the model is compiled.

---

## Model family

For every observed cell in `train + test + arc-gen[:100]`, fit:

```text
output_color = f(input_color)
```

Reject the solver if:

- input/output shapes differ
- one input color maps to multiple output colors
- the compiled ONNX fails full validation

Compile the accepted map as the existing 1x1 `Conv` color remap.

---

## Why this is the right next lever

submission-4 established the pattern:

```text
symbolic family + ARC-GEN fitting + strict validation => Kaggle delta
```

Color maps are the smallest next symbolic family. They add fitting evidence without adding compiler risk.

---

## Expected failure mode

The likely result is zero new tasks. That is acceptable if the harness proves:

- phase 16 keeps `arcgen_validate_samples >= 100`
- no train_only outputs are included
- submit gate blocks non-improvements

---

## Next theory if Phase 16 is flat

Move from color-only parameter fitting to **layout-relative fitting**:

```text
runtime bbox -> relative source index -> gather output
```

That is a stronger Code World Model lever, but it needs a dynamic ONNX index compiler and should follow the small Phase 16 probe.

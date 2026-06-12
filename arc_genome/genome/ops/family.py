"""Phase 3 parameterized family solvers."""

from __future__ import annotations

import numpy as np

from arc_genome.data.encoding import fixed_shapes, get_examples
from arc_genome.genome.ops.analytical import s_color_map
from arc_genome.genome.ops.extended import s_translate
from arc_genome.onnx.gather import build_gather_model


def s_color_then_translate(td):
    """Compose color map then spatial translate."""
    cm_model = s_color_map(td)
    if cm_model is None:
        return None
    # Simulate color map on numpy grids
    exs = get_examples(td)
    mapped = []
    colors = {}
    for inp, out in exs:
        for iv, ov in zip(inp.flat, out.flat):
            colors[int(iv)] = int(ov)
    for inp, out in exs:
        m = inp.copy()
        for c in np.unique(inp):
            if int(c) in colors:
                m[inp == c] = colors[int(c)]
        mapped.append((m, out))
    fake_td = {"train": [], "test": []}
    for i, (m, o) in enumerate(mapped):
        ex = {"input": m.tolist(), "output": o.tolist()}
        if i < len(td["train"]):
            fake_td["train"].append(ex)
        else:
            fake_td["test"].append(ex)
    return s_translate(fake_td)


def s_extract_and_recolor(td):
    """If output is uniform recolor of nonzero region shape change."""
    sp = fixed_shapes(td)
    if sp is None:
        return None
    (ih, iw), (oh, ow) = sp
    exs = get_examples(td)
    target_colors = set()
    for _, out in exs:
        nz = out[out != 0]
        if len(set(nz.tolist())) == 1:
            target_colors.add(int(nz[0]) if len(nz) else 0)
    if len(target_colors) != 1:
        return None
    tc = target_colors.pop()
    for inp, out in exs:
        expected = np.zeros_like(out)
        mask = inp != 0
        if mask.shape != out.shape:
            return None
        expected[mask] = tc
        if not np.array_equal(expected, out):
            return None
    return s_color_map(td)


FAMILY_SOLVERS = [
    ("color_then_translate", s_color_then_translate),
    ("extract_recolor", s_extract_and_recolor),
]

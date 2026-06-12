"""Content-aware example normalization (phase 2)."""

from __future__ import annotations

import copy

import numpy as np


def _bbox(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = grid != 0
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def normalize_task_data(task_data: dict) -> dict:
    """Crop all examples to a shared tight bounding box."""
    exs = task_data["train"] + task_data["test"]
    boxes = []
    for ex in exs:
        for key in ("input", "output"):
            g = np.array(ex[key])
            b = _bbox(g)
            if b:
                boxes.append(b)

    if not boxes:
        return task_data

    r0 = min(b[0] for b in boxes)
    r1 = max(b[1] for b in boxes)
    c0 = min(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)

    norm = copy.deepcopy(task_data)
    for split in ("train", "test"):
        for ex in norm[split]:
            for key in ("input", "output"):
                g = np.array(ex[key])
                ex[key] = g[r0:r1, c0:c1].tolist()
    return norm

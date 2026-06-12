"""Grid encoding and task loading."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

BATCH, CH, GH, GW = 1, 10, 30, 30
GRID_SHAPE = [BATCH, CH, GH, GW]


def load_tasks_json(data_file: str) -> dict[int, dict[str, Any]]:
    with open(data_file) as f:
        raw = json.load(f)
    tasks: dict[int, dict[str, Any]] = {}
    for i, (task_id, task_data) in enumerate(sorted(raw.items()), 1):
        tasks[i] = {"hex": task_id, "data": task_data}
    return tasks


def to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, CH, GH, GW), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            arr[0, v, r, c] = 1.0
    return arr


def get_examples(task_data: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.array(ex["input"], dtype=np.int64), np.array(ex["output"], dtype=np.int64))
        for ex in task_data["train"] + task_data["test"]
    ]


def fixed_shapes(task_data: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    shapes: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for inp, out in get_examples(task_data):
        shapes.add((tuple(inp.shape), tuple(out.shape)))
    return list(shapes)[0] if len(shapes) == 1 else None

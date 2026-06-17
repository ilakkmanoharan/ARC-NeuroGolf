"""ARC-GEN synthetic validation."""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort

from arc_genome.config import get_config
from arc_genome.data.encoding import to_onehot

ORT_PROVIDERS = ["CPUExecutionProvider"]
MAX_ARCGEN_VALIDATE = 30  # Kaggle official cap
_arcgen_dir: str | None = None
_tasks_cache: dict | None = None


def set_arcgen_dir(path: str = "data/arc_gen_raw") -> None:
    global _arcgen_dir, _tasks_cache
    _arcgen_dir = path
    _tasks_cache = None


def attach_arcgen(task_data: dict, task_hex: str, arcgen_dir: str = "data/arc_gen_raw") -> dict:
    """Return task_data copy with arc-gen examples attached."""
    out = dict(task_data)
    path = os.path.join(arcgen_dir, f"{task_hex}.json")
    if os.path.exists(path):
        with open(path) as f:
            examples = json.load(f)
        if isinstance(examples, list):
            out["arc-gen"] = examples
        else:
            out["arc-gen"] = []
    else:
        out["arc-gen"] = []
    return out


def load_tasks_with_arcgen(data_file: str = "data/all_tasks.json", arcgen_dir: str = "data/arc_gen_raw") -> dict:
    global _tasks_cache
    if _tasks_cache is not None:
        return _tasks_cache
    with open(data_file) as f:
        raw = json.load(f)
    tasks: dict = {}
    for i, (hex_id, task_data) in enumerate(sorted(raw.items()), 1):
        td = attach_arcgen(task_data, hex_id, arcgen_dir)
        tasks[i] = {"hex": hex_id, "data": td}
    _tasks_cache = tasks
    return tasks


def _arcgen_limit() -> int:
    cfg = get_config()
    if cfg.arcgen_validation:
        return cfg.arcgen_validate_samples
    return MAX_ARCGEN_VALIDATE


def validate_full(path: str, task_data: dict) -> bool:
    """Validate train + test + arc-gen (configurable sample count)."""
    try:
        sess = ort.InferenceSession(path, providers=ORT_PROVIDERS)
    except Exception:
        return False
    n = _arcgen_limit()
    examples = task_data.get("train", []) + task_data.get("test", []) + task_data.get("arc-gen", [])[:n]
    for ex in examples:
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        try:
            out = sess.run(["output"], {"input": inp})[0]
            out = (out > 0.0).astype(np.float32)
        except Exception:
            return False
        if not np.array_equal(out, exp):
            return False
    return True

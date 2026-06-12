"""Smoke tests for ARC-Genome package."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.data.encoding import to_onehot
from arc_genome.genome.ops.analytical import s_identity
from arc_genome.onnx.cost import compute_cost
from arc_genome.onnx.model import save_model, validate_model


def test_identity_solver():
    td = {
        "train": [{"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]}],
        "test": [{"input": [[2, 3], [3, 2]], "output": [[2, 3], [3, 2]]}],
    }
    model = s_identity(td)
    assert model is not None

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "task001.onnx")
        save_model(model, path)
        assert validate_model(path, td)
        cost = compute_cost(path)
        assert cost["total"] > 0


def test_to_onehot_shape():
    grid = [[0, 1], [2, 3]]
    arr = to_onehot(grid)
    assert arr.shape == (1, 10, 30, 30)
    assert arr[0, 1, 0, 1] == 1.0

#!/usr/bin/env python3
"""Build probe ONNX models for phase 1 cost calibration."""

from __future__ import annotations

import os
import sys

import numpy as np
from onnx import helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.onnx.cost import compute_cost
from arc_genome.onnx.model import make_model, save_model

OUT = "phases/phase1/probes"
os.makedirs(OUT, exist_ok=True)


def probe_identity():
    return make_model([helper.make_node("Identity", ["input"], ["output"])])


def probe_color_map():
    w = np.eye(10, dtype=np.float32).reshape(10, 10, 1, 1)
    return make_model(
        [helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1])],
        [numpy_helper.from_array(w, "W")],
    )


def probe_conv3():
    w = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for i in range(10):
        w[i, i, 1, 1] = 1.0
    return make_model(
        [helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        [numpy_helper.from_array(w, "W")],
    )


def main():
    probes = {"identity": probe_identity(), "color_map_1x1": probe_color_map(), "conv_3x3": probe_conv3()}
    for name, model in probes.items():
        path = os.path.join(OUT, f"{name}.onnx")
        save_model(model, path)
        cost = compute_cost(path)
        print(f"{name}: cost={cost['total']:,} score={cost['score']:.2f}")


if __name__ == "__main__":
    main()

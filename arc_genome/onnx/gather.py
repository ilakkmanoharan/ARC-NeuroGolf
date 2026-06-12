"""Gather-based spatial remapping ONNX graphs."""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper

from arc_genome.data.encoding import GH, GW
from arc_genome.onnx.model import make_model


def build_gather_model(oh: int, ow: int, idx: np.ndarray):
    flat_idx = np.zeros((1, 10, GH * GW), dtype=np.int64)
    mask = np.zeros((1, 1, GH, GW), dtype=np.float32)
    for oi in range(oh):
        for oj in range(ow):
            flat_idx[0, :, oi * GW + oj] = idx[oi, oj, 0] * GW + idx[oi, oj, 1]
            mask[0, 0, oi, oj] = 1.0
    inits = [
        numpy_helper.from_array(np.array([1, 10, GH * GW], dtype=np.int64), "fs"),
        numpy_helper.from_array(flat_idx, "idx"),
        numpy_helper.from_array(np.array([1, 10, GH, GW], dtype=np.int64), "os"),
        numpy_helper.from_array(mask, "mask"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "fs"], ["flat"]),
        helper.make_node("GatherElements", ["flat", "idx"], ["g"], axis=2),
        helper.make_node("Reshape", ["g", "os"], ["raw"]),
        helper.make_node("Mul", ["raw", "mask"], ["output"]),
    ]
    return make_model(nodes, inits)


def build_gather_model_with_const(ih: int, iw: int, oh: int, ow: int, idx: np.ndarray, cst: np.ndarray):
    flat_idx = np.zeros((1, 10, GH * GW), dtype=np.int64)
    gather_mask = np.zeros((1, 1, GH, GW), dtype=np.float32)
    const_oh = np.zeros((1, 10, GH, GW), dtype=np.float32)
    for oi in range(oh):
        for oj in range(ow):
            if idx[oi, oj, 0] >= 0:
                flat_idx[0, :, oi * GW + oj] = idx[oi, oj, 0] * GW + idx[oi, oj, 1]
                gather_mask[0, 0, oi, oj] = 1.0
            elif cst[oi, oj] >= 0:
                const_oh[0, cst[oi, oj], oi, oj] = 1.0
    has_const = np.any(const_oh > 0)
    inits = [
        numpy_helper.from_array(np.array([1, 10, GH * GW], dtype=np.int64), "fs"),
        numpy_helper.from_array(flat_idx, "idx"),
        numpy_helper.from_array(np.array([1, 10, GH, GW], dtype=np.int64), "os"),
        numpy_helper.from_array(gather_mask, "gmask"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "fs"], ["flat"]),
        helper.make_node("GatherElements", ["flat", "idx"], ["g"], axis=2),
        helper.make_node("Reshape", ["g", "os"], ["raw"]),
        helper.make_node("Mul", ["raw", "gmask"], ["masked"]),
    ]
    if has_const:
        inits.append(numpy_helper.from_array(const_oh, "cst"))
        nodes.append(helper.make_node("Add", ["masked", "cst"], ["output"]))
    else:
        nodes[-1] = helper.make_node("Mul", ["raw", "gmask"], ["output"])
    return make_model(nodes, inits)

"""ONNX model construction and validation."""

from __future__ import annotations

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from arc_genome.data.encoding import GRID_SHAPE, to_onehot

DT = TensorProto.FLOAT
IR = 10
OPSET = [helper.make_opsetid("", 10)]
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}
ORT_PROVIDERS = ["CPUExecutionProvider"]


def make_model(nodes, inits=None):
    x = helper.make_tensor_value_info("input", DT, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DT, GRID_SHAPE)
    g = helper.make_graph(nodes, "g", [x], [y], initializer=inits or [])
    return helper.make_model(g, ir_version=IR, opset_imports=OPSET)


def check_banned_ops(model) -> list[str]:
    return [n.op_type for n in model.graph.node if n.op_type in BANNED_OPS]


def validate_model(path: str, task_data: dict) -> bool:
    try:
        model = onnx.load(path)
        if check_banned_ops(model):
            return False
        sess = ort.InferenceSession(path, providers=ORT_PROVIDERS)
    except Exception:
        return False

    for ex in task_data["train"] + task_data["test"]:
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        try:
            out = sess.run(["output"], {"input": inp})[0]
            out = (out > 0.0).astype("float32")
        except Exception:
            return False
        if not (out == exp).all():
            return False
    return True


def save_model(model, path: str) -> None:
    onnx.save(model, path)

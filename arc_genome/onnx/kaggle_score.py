"""Official Kaggle cost/score (ported from rogermt/neurogolf-solver medal-solvers/score_model.py)."""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

_EXCLUDED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sanitize_model(model: onnx.ModelProto):
    for node in model.graph.node:
        node.name = node.output[0] if node.output else ""
        if node.output and "kernel_time" in node.output[0]:
            return None
    name_map: dict[str, str] = {}
    counter = 0

    def safe(old: str) -> str:
        nonlocal counter
        if not old or old in ("input", "output"):
            return old
        if old not in name_map:
            name_map[old] = f"safe_name_{counter}"
            counter += 1
        return name_map[old]

    for inp in model.graph.input:
        inp.name = safe(inp.name)
    for init in model.graph.initializer:
        init.name = safe(init.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = safe(node.input[i])
        for i in range(len(node.output)):
            node.output[i] = safe(node.output[i])
        if node.output and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = safe(out.name)
    for vi in model.graph.value_info:
        vi.name = safe(vi.name)
    for node in model.graph.node:
        node.name = node.output[0] if node.output else ""
    return model


def _calculate_memory(model, trace_path: str) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None
        node_outputs[node.name] = list(node.output)
        for o in node.output:
            if o:
                tensor_names.add(o)

    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype] = {}
    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
    tensor_names.update(tensor_map.keys())

    for tname in tensor_names:
        item = tensor_map.get(tname)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tt = item.type.tensor_type
        if not tt.HasField("shape"):
            return None
        num_el = 1
        for dim in tt.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num_el *= dim.dim_value
        if tname in ("input", "output"):
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        tensor_memory[tname] = num_el * np.dtype(np_dtype).itemsize
        tensor_dtypes[tname] = np_dtype

    seen: set[str] = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)

    for node in graph.node:
        for o in node.output:
            if o and o != "output":
                item = tensor_map.get(o)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    with open(trace_path) as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory.get(output_name, 0), mem)
    return sum(tensor_memory.values())


def _calculate_params(model) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(d <= 0 for d in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d <= 0 for d in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                if any(d <= 0 for d in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def kaggle_score_model(model_path: str, task_data: dict) -> dict | None:
    """Return {score, memory, params, cost} matching Kaggle evaluator."""
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.op_type.upper() in _EXCLUDED:
            return None
        if "Sequence" in node.op_type:
            return None

    sanitized = sanitize_model(model)
    if sanitized is None:
        return None

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    prefix = os.path.join(tempfile.gettempdir(), f"arc_{os.getpid()}_{id(model_path)}")
    options.profile_file_prefix = prefix

    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])
    except Exception:
        return None

    examples = task_data.get("train", []) + task_data.get("test", []) + task_data.get("arc-gen", [])[:30]
    for ex in examples:
        grid = ex["input"]
        if max(len(grid), max((len(r) for r in grid), default=0)) > 30:
            continue
        inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if r < 30 and c < 30:
                    inp[0, v, r, c] = 1.0
        try:
            session.run(["output"], {"input": inp})
        except Exception:
            pass

    trace_path = session.end_profiling()
    try:
        memory = _calculate_memory(sanitized, trace_path)
        params = _calculate_params(sanitized)
    finally:
        if trace_path and os.path.exists(trace_path):
            os.remove(trace_path)

    if memory is None or params is None or memory < 0 or params < 0:
        return None
    cost = memory + params
    score = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return {"score": score, "memory": memory, "params": params, "cost": cost}

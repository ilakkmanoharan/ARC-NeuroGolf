"""Cost estimation: params + memory bytes + MACs (phase-1 calibrated)."""

from __future__ import annotations

import math

import onnx
from onnx import numpy_helper

DTYPE_BYTES = {1: 4, 6: 4, 7: 8, 11: 8, 12: 8}


def _tensor_bytes(tensor) -> int:
    if tensor.HasField("raw_data"):
        return len(tensor.raw_data)
    elem_type = tensor.data_type
    bpe = DTYPE_BYTES.get(elem_type, 4)
    if tensor.float_data:
        return len(tensor.float_data) * bpe
    if tensor.int64_data:
        return len(tensor.int64_data) * bpe
    if tensor.int32_data:
        return len(tensor.int32_data) * bpe
    arr = numpy_helper.to_array(tensor)
    return int(arr.nbytes)


def _num_params(model) -> int:
    return sum(numpy_helper.to_array(init).size for init in model.graph.initializer)


def _memory_bytes(model) -> int:
    seen: set[str] = set()
    total = 0
    for init in model.graph.initializer:
        total += _tensor_bytes(init)
        seen.add(init.name)
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name in seen:
            continue
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        elems = 1
        for d in shape:
            if d > 0:
                elems *= d
        total += elems * 4
    return total


def _volume(shape: list[int]) -> int:
    v = 1
    for d in shape:
        if d > 0:
            v *= d
    return v


def _get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
    return default


def _infer_shapes(model) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for inp in model.graph.input:
        shapes[inp.name] = [d.dim_value for d in inp.type.tensor_type.shape.dim]

    for node in model.graph.node:
        ins = [shapes.get(i, [1, 10, 30, 30]) for i in node.input if i]
        in_shape = ins[0] if ins else [1, 10, 30, 30]
        op = node.op_type
        out_shape = list(in_shape)

        if op == "Conv":
            kh, kw = (_get_attr(node, "kernel_shape", [1, 1]) or [1, 1])[:2]
            pads = _get_attr(node, "pads", [0, 0, 0, 0]) or [0, 0, 0, 0]
            strides = _get_attr(node, "strides", [1, 1]) or [1, 1]
            _, cin, hin, win = in_shape if len(in_shape) == 4 else [1, 10, 30, 30]
            hout = (hin + pads[0] + pads[2] - kh) // strides[0] + 1
            wout = (win + pads[1] + pads[3] - kw) // strides[1] + 1
            cout = 10
            if len(node.input) > 1:
                for init in model.graph.initializer:
                    if init.name == node.input[1]:
                        w = numpy_helper.to_array(init)
                        cout = w.shape[0]
                        break
            out_shape = [1, cout, hout, wout]

        elif op == "Slice":
            starts = ends = None
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    starts = numpy_helper.to_array(init).tolist()
                if len(node.input) > 2 and init.name == node.input[2]:
                    ends = numpy_helper.to_array(init).tolist()
            if starts and ends:
                out_shape = [
                    ends[i] - starts[i] if i < len(ends) else in_shape[i]
                    for i in range(len(in_shape))
                ]

        elif op == "Pad":
            pads = _get_attr(node, "pads", [0] * 8) or [0] * 8
            if len(in_shape) == 4:
                out_shape = [
                    in_shape[0],
                    in_shape[1],
                    in_shape[2] + pads[2] + pads[6],
                    in_shape[3] + pads[3] + pads[7],
                ]

        elif op == "Transpose":
            perm = _get_attr(node, "perm", [0, 1, 2, 3]) or [0, 1, 2, 3]
            out_shape = [in_shape[p] for p in perm]

        elif op == "Reshape":
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    out_shape = numpy_helper.to_array(init).tolist()
                    break

        elif op == "ArgMax":
            axis = _get_attr(node, "axis", 1)
            keepdims = _get_attr(node, "keepdims", 1)
            out_shape = list(in_shape)
            if axis < len(out_shape):
                if not keepdims:
                    out_shape.pop(axis)
                else:
                    out_shape[axis] = 1

        elif op == "ReduceSum":
            axes = _get_attr(node, "axes", [1]) or [1]
            keepdims = _get_attr(node, "keepdims", 0)
            out_shape = list(in_shape)
            for ax in sorted(axes, reverse=True):
                if ax < len(out_shape):
                    if keepdims:
                        out_shape[ax] = 1
                    else:
                        out_shape.pop(ax)

        elif op == "OneHot":
            out_shape = [1, 10, in_shape[2], in_shape[3]] if len(in_shape) >= 4 else [1, 10, 30, 30]

        elif op in ("Mul", "Add", "Sub", "Div", "GatherElements"):
            if len(ins) > 1 and len(ins[1]) == len(in_shape):
                out_shape = list(in_shape)
            else:
                out_shape = list(in_shape)

        elif op == "Tile":
            reps = None
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    reps = numpy_helper.to_array(init).tolist()
            if reps and len(reps) == len(in_shape):
                out_shape = [in_shape[i] * reps[i] for i in range(len(in_shape))]

        elif op == "Identity":
            out_shape = list(in_shape)

        shapes[node.output[0]] = out_shape

    return shapes


def _estimate_macs(model) -> int:
    shapes = _infer_shapes(model)
    macs = 0
    for node in model.graph.node:
        op = node.op_type
        in_shape = shapes.get(node.input[0], [1, 10, 30, 30])
        out_shape = shapes.get(node.output[0], in_shape)
        vol_in = _volume(in_shape)
        vol_out = _volume(out_shape)

        if op == "Conv":
            kh, kw = (_get_attr(node, "kernel_shape", [1, 1]) or [1, 1])[:2]
            cout = out_shape[1] if len(out_shape) > 1 else 10
            hout = out_shape[2] if len(out_shape) > 2 else 30
            wout = out_shape[3] if len(out_shape) > 3 else 30
            cin = in_shape[1] if len(in_shape) > 1 else 10
            macs += cout * hout * wout * cin * kh * kw
        elif op in ("MatMul", "Gemm"):
            macs += vol_out * 10
        elif op in ("Add", "Mul", "Sub", "Div"):
            macs += max(vol_in, vol_out)
        elif op == "ArgMax":
            macs += vol_in
        elif op == "OneHot":
            macs += vol_out
        elif op == "GatherElements":
            macs += vol_out * 2
        elif op == "ReduceSum":
            macs += vol_in
        elif op in ("Slice", "Pad", "Transpose", "Reshape", "Tile", "Identity"):
            macs += max(vol_in, vol_out) // 10
    return int(macs)


def compute_cost(model_path: str) -> dict[str, int]:
    model = onnx.load(model_path)
    params = _num_params(model)
    mem = _memory_bytes(model)
    macs = _estimate_macs(model)
    total = params + mem + macs
    return {
        "params": params,
        "memory_bytes": mem,
        "macs": macs,
        "total": total,
        "score": compute_score(total),
    }


def compute_score(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(cost, 1)))

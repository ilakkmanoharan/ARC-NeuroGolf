"""Per-task genome inference orchestrator."""

from __future__ import annotations

import os
import shutil
import tempfile

from arc_genome.config import get_config
from arc_genome.data.arcgen import validate_full
from arc_genome.data.normalize import normalize_task_data
from arc_genome.genome.ops.analytical import ANALYTICAL_SOLVERS
from arc_genome.genome.ops.conv import (
    solve_conv_diffshape,
    solve_conv_fixed,
    solve_conv_variable,
    solve_conv_v2,
)
from arc_genome.data.encoding import fixed_shapes, get_examples
from arc_genome.onnx.cost import compute_cost
from arc_genome.onnx.model import save_model, validate_model


def _solver_list():
    cfg = get_config()
    solvers = list(ANALYTICAL_SOLVERS)
    if cfg.extended_analytical:
        from arc_genome.genome.ops.extended import EXTENDED_SOLVERS
        solvers.extend(EXTENDED_SOLVERS)
    if cfg.family_solvers:
        from arc_genome.genome.ops.family import FAMILY_SOLVERS
        solvers.extend(FAMILY_SOLVERS)
    return solvers


def _validates(path: str, task_data: dict, task_hex: str | None = None) -> bool:
    cfg = get_config()
    if cfg.arcgen_validation:
        return validate_full(path, task_data)
    return validate_model(path, task_data)


def _try_record(
    candidates: list,
    path: str,
    sname: str,
    task_data: dict,
    task_hex: str | None,
    apply_gate: bool,
):
    if not _validates(path, task_data, task_hex):
        return
    cost = compute_cost(path)
    cfg = get_config()
    if apply_gate and cfg.calibrated_cost and cost["score"] < cfg.cost_gate_min_score:
        return
    candidates.append((sname, cost, path))


def _gather_candidates(
    task_data: dict,
    task_hex: str | None,
    tmpdir: str,
    conv_budget: float,
    apply_gate: bool,
) -> list:
    cfg = get_config()
    candidates: list = []
    datasets = [task_data]
    if cfg.content_normalize:
        datasets.append(normalize_task_data(task_data))
    idx = 0

    if cfg.composition_depth > 0:
        from arc_genome.genome.compose.search import solve_composition
        for td in datasets:
            try:
                model = solve_composition(td, cfg.composition_depth)
                if model is None:
                    continue
                p = os.path.join(tmpdir, f"c{idx}.onnx")
                idx += 1
                save_model(model, p)
                _try_record(candidates, p, "compose", td, task_hex, apply_gate)
            except Exception:
                pass

    for td in datasets:
        for sname, sfn in _solver_list():
            try:
                model = sfn(td)
                if model is None:
                    continue
                p = os.path.join(tmpdir, f"c{idx}.onnx")
                idx += 1
                save_model(model, p)
                _try_record(candidates, p, sname, td, task_hex, apply_gate)
            except Exception:
                continue

    return candidates


def _gather_conv(
    task_data: dict,
    task_hex: str | None,
    tmpdir: str,
    conv_budget: float,
    apply_gate: bool,
    idx_start: int = 0,
) -> tuple[list, int]:
    cfg = get_config()
    candidates: list = []
    idx = idx_start
    exs = get_examples(task_data)
    same_shape = all(inp.shape == out.shape for inp, out in exs)
    shapes = {inp.shape for inp, _ in exs}
    fixed_in = len(shapes) == 1
    mk = cfg.max_kernel

    attempts: list[tuple[str, object]] = []
    if same_shape:
        if fixed_in:
            attempts.append(
                ("conv_fixed", lambda p: solve_conv_fixed(task_data, p, conv_budget, mk))
            )
        attempts.append(("conv_var", lambda p: solve_conv_variable(task_data, p, conv_budget, mk)))
    else:
        sp = fixed_shapes(task_data)
        if sp is not None:
            (ih, iw), (oh, ow) = sp
            if oh <= ih and ow <= iw:
                attempts.append(
                    ("conv_diff", lambda p: solve_conv_diffshape(task_data, p, conv_budget, mk))
                )
    if cfg.conv_v2:
        attempts.append(("conv_v2", lambda p: solve_conv_v2(task_data, p, cfg.unsolved_budget)))

    for sname, fn in attempts:
        p = os.path.join(tmpdir, f"c{idx}.onnx")
        idx += 1
        try:
            if fn(p) is None:
                continue
            _try_record(candidates, p, sname, task_data, task_hex, apply_gate)
        except Exception:
            continue
    return candidates, idx


def solve_task(
    task_num: int,
    task_data: dict,
    outdir: str,
    conv_budget: float = 30.0,
    task_hex: str | None = None,
) -> tuple[bool, str | None, dict | None]:
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"task{task_num:03d}.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        candidates = _gather_candidates(task_data, task_hex, tmpdir, conv_budget, apply_gate=True)
        if not candidates:
            conv_cands, _ = _gather_conv(task_data, task_hex, tmpdir, conv_budget, apply_gate=True)
            candidates = conv_cands
        # Among analytical hits, still allow cheaper conv if analytical only option is expensive
        if candidates and all(c[1]["score"] < 12 for c in candidates):
            conv_cands, _ = _gather_conv(task_data, task_hex, tmpdir, conv_budget, apply_gate=True)
            candidates.extend(conv_cands)
        if not candidates:
            candidates = _gather_candidates(task_data, task_hex, tmpdir, conv_budget, apply_gate=False)
            if not candidates:
                conv_cands, _ = _gather_conv(task_data, task_hex, tmpdir, conv_budget, apply_gate=False)
                candidates = conv_cands
        if not candidates:
            if os.path.exists(out_path):
                os.remove(out_path)
            return False, None, None
        name, cost, src = min(candidates, key=lambda x: x[1]["total"])
        shutil.copy2(src, out_path)

    return True, name, cost


def cost_audit_task(
    task_num: int,
    task_data: dict,
    outdir: str,
    conv_budget: float,
    task_hex: str | None = None,
) -> tuple[bool, str | None, dict | None]:
    out_path = os.path.join(outdir, f"task{task_num:03d}.onnx")
    old_cost = compute_cost(out_path) if os.path.exists(out_path) else None
    ok, name, cost = solve_task(task_num, task_data, outdir, conv_budget, task_hex)
    if not ok:
        return False, None, None
    if old_cost and cost["total"] >= old_cost["total"]:
        return True, name, old_cost
    return ok, name, cost

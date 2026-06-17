"""Batch solving and submission packaging."""

from __future__ import annotations

import os
import shutil
import time
import zipfile
from collections import Counter

from arc_genome.config import get_config
from arc_genome.data.arcgen import load_tasks_with_arcgen
from arc_genome.data.encoding import load_tasks_json
from arc_genome.genome.infer import cost_audit_task, solve_task
from arc_genome.onnx.cost import compute_cost, compute_score


def solve_all(
    data_file: str,
    output_dir: str = "submission",
    conv_budget: float = 30.0,
    task_nums: list[int] | None = None,
) -> dict:
    tasks = load_tasks_with_arcgen(data_file)
    if task_nums is None:
        task_nums = sorted(tasks.keys())

    cfg = get_config()
    results: dict[int, dict] = {}
    t0 = time.time()
    solve_fn = cost_audit_task if cfg.cost_audit else solve_task

    for tn in task_nums:
        if tn not in tasks:
            continue
        td = tasks[tn]["data"]
        hex_id = tasks[tn].get("hex")
        ok, sname, cost_info = solve_fn(tn, td, output_dir, conv_budget, hex_id)
        if ok:
            results[tn] = {"solver": sname, "cost": cost_info}
            print(
                f"Task {tn:3d}: {sname:20s} "
                f"cost={cost_info['total']:>10,} score={cost_info.get('score', compute_score(cost_info['total'])):.1f}"
            )
        else:
            print(f"Task {tn:3d}: UNSOLVED")

    if cfg.second_pass:
        unsolved = [tn for tn in task_nums if tn not in results]
        print(f"\n--- Second pass: {len(unsolved)} unsolved tasks ---")
        for tn in unsolved:
            td = tasks[tn]["data"]
            hex_id = tasks[tn].get("hex")
            ok, sname, cost_info = solve_task(tn, td, output_dir, cfg.unsolved_budget, hex_id)
            if ok:
                results[tn] = {"solver": sname, "cost": cost_info}
                print(f"Task {tn:3d}: {sname:20s} (2nd pass) cost={cost_info['total']:>10,}")

    if cfg.third_pass:
        unsolved = [tn for tn in task_nums if tn not in results]
        print(f"\n--- Third pass (conv_v2 extended): {len(unsolved)} unsolved tasks ---")
        from arc_genome.genome.ops.conv import solve_conv_v2
        import tempfile
        for tn in unsolved:
            td = tasks[tn]["data"]
            hex_id = tasks[tn].get("hex")
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "m.onnx")
                if solve_conv_v2(td, path, cfg.third_pass_budget) is None:
                    continue
                from arc_genome.genome.infer import _validates, _score_candidate
                if not _validates(path, td, hex_id):
                    continue
                cost_info = _score_candidate(path, td)
                out_path = os.path.join(output_dir, f"task{tn:03d}.onnx")
                shutil.copy2(path, out_path)
                results[tn] = {"solver": "conv_v2", "cost": cost_info}
                print(f"Task {tn:3d}: conv_v2              (3rd pass) cost={cost_info['total']:>10,}")

    elapsed = time.time() - t0
    total_score = sum(r["cost"].get("score", compute_score(r["cost"]["total"])) for r in results.values())
    print(f"\nSolved: {len(results)}/{len(task_nums)} in {elapsed:.0f}s")
    print(f"Estimated total score: {total_score:.1f}")
    sc = Counter(r["solver"] for r in results.values())
    for s, c in sc.most_common():
        print(f"  {s}: {c}")

    return results


def make_submission_zip(
    output_dir: str = "submission",
    zip_path: str = "submission.zip",
    data_file: str | None = None,
) -> str:
    from arc_genome.config import get_config
    from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
    from arc_genome.onnx.model import validate_model

    tasks = load_tasks_with_arcgen(data_file) if data_file else None
    cfg = get_config()
    included = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(output_dir)):
            if not f.endswith(".onnx"):
                continue
            path = os.path.join(output_dir, f)
            if tasks is not None:
                task_num = int(f.replace("task", "").replace(".onnx", ""))
                if task_num not in tasks:
                    continue
                td = tasks[task_num]["data"]
                ok = validate_full(path, td) if cfg.arcgen_validation else validate_model(path, td)
                if not ok:
                    continue
            zf.write(path, f)
            included += 1
    size_kb = os.path.getsize(zip_path) / 1024
    print(f"Created {zip_path}: {included} files, {size_kb:.0f} KB")
    return zip_path

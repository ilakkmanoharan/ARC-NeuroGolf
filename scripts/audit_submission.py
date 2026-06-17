#!/usr/bin/env python3
"""Audit submission ONNX files: validation tiers + official Kaggle score."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.data.arcgen import attach_arcgen, load_tasks_with_arcgen, validate_full
from arc_genome.data.encoding import load_tasks_json
from arc_genome.onnx.cost import compute_cost
from arc_genome.onnx.kaggle_score import kaggle_score_model
from arc_genome.onnx.model import validate_model


def audit(submission_dir: str, data_file: str, out_json: str, phase: int = 6):
    from arc_genome.config import set_phase
    set_phase(phase)
    tasks = load_tasks_with_arcgen(data_file)
    results = []
    buckets = {"pass_all": 0, "train_only": 0, "fail": 0}

    for tn in sorted(tasks.keys()):
        path = os.path.join(submission_dir, f"task{tn:03d}.onnx")
        if not os.path.exists(path):
            continue
        hex_id = tasks[tn]["hex"]
        td = tasks[tn]["data"]
        train_ok = validate_model(path, {"train": td["train"], "test": td["test"]})
        full_ok = validate_full(path, td) if train_ok else False

        if full_ok:
            buckets["pass_all"] += 1
            tier = "pass_all"
        elif train_ok:
            buckets["train_only"] += 1
            tier = "train_only"
        else:
            buckets["fail"] += 1
            tier = "fail"

        local = compute_cost(path)
        kaggle = kaggle_score_model(path, td) if train_ok else None

        results.append({
            "task": tn,
            "hex": hex_id,
            "tier": tier,
            "local_score": local["score"],
            "local_cost": local["total"],
            "kaggle_score": kaggle["score"] if kaggle else None,
            "kaggle_cost": kaggle["cost"] if kaggle else None,
            "kaggle_memory": kaggle["memory"] if kaggle else None,
            "kaggle_params": kaggle["params"] if kaggle else None,
        })
        if tn % 50 == 0:
            print(f"  audited {tn}...")

    kaggle_scores = [r["kaggle_score"] for r in results if r["kaggle_score"]]
    local_scores = [r["local_score"] for r in results]

    summary = {
        "submission_dir": submission_dir,
        "files_audited": len(results),
        "buckets": buckets,
        "local_total_score": sum(local_scores),
        "kaggle_total_score": sum(kaggle_scores),
        "kaggle_avg": sum(kaggle_scores) / len(kaggle_scores) if kaggle_scores else 0,
        "pass_all_kaggle_total": sum(r["kaggle_score"] or 0 for r in results if r["tier"] == "pass_all"),
    }

    report = {"summary": summary, "tasks": results}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Audit Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", default="phases/phase6/submission")
    parser.add_argument("--data_file", default="data/all_tasks.json")
    parser.add_argument("--out", default="phases/milestone1/audit.json")
    parser.add_argument("--phase", type=int, default=6)
    args = parser.parse_args()
    audit(args.submission_dir, args.data_file, args.out, args.phase)


if __name__ == "__main__":
    main()

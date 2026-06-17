#!/usr/bin/env python3
"""Download official Kaggle task JSONs and build per-task scoring logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.data.arcgen import load_tasks_with_arcgen
from arc_genome.onnx.kaggle_score import kaggle_score_model
from arc_genome.onnx.model import validate_model
from arc_genome.data.arcgen import validate_full

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))


def download_task_json(task_num: int, out_dir: str) -> str | None:
    fname = f"task{task_num:03d}.json"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        return path
    proc = subprocess.run(
        [KAGGLE, "competitions", "download", "-c", "neurogolf-2026", "-f", fname, "-p", out_dir, "--force"],
        capture_output=True, text=True,
    )
    dl = os.path.join(out_dir, "DownloadDataFile")
    if os.path.exists(dl) and not os.path.exists(path):
        os.rename(dl, path)
    return path if os.path.exists(path) else None


def score_task(task_num: int, onnx_path: str, td: dict) -> dict:
    train_ok = validate_model(onnx_path, {"train": td["train"], "test": td["test"]})
    full_ok = validate_full(onnx_path, td) if train_ok else False
    ks = kaggle_score_model(onnx_path, td) if train_ok else None
    return {
        "task": task_num,
        "train_ok": train_ok,
        "pass_all": full_ok,
        "kaggle_score": ks["score"] if ks else None,
        "kaggle_memory": ks["memory"] if ks else None,
        "kaggle_params": ks["params"] if ks else None,
        "kaggle_cost": ks["cost"] if ks else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_file", default="data/all_tasks.json")
    parser.add_argument("--download_official", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    official_dir = os.path.join(args.out_dir, "official_tasks")
    if args.download_official:
        os.makedirs(official_dir, exist_ok=True)

    tasks = load_tasks_with_arcgen(args.data_file)
    logs = []
    total = 0.0
    for f in sorted(os.listdir(args.submission_dir)):
        if not f.endswith(".onnx"):
            continue
        tn = int(f.replace("task", "").replace(".onnx", ""))
        if tn not in tasks:
            continue
        td = tasks[tn]["data"]
        onnx_path = os.path.join(args.submission_dir, f)
        entry = score_task(tn, onnx_path, td)
        if args.download_official:
            off = download_task_json(tn, official_dir)
            entry["official_json"] = off
        logs.append(entry)
        if entry["kaggle_score"]:
            total += entry["kaggle_score"]

    summary = {
        "submission_dir": args.submission_dir,
        "tasks_scored": len(logs),
        "pass_all": sum(1 for e in logs if e["pass_all"]),
        "kaggle_total_est": total,
        "kaggle_actual": 848.07,
        "audit_ratio": 848.07 / total if total else None,
    }
    out = {"summary": summary, "tasks": logs}
    with open(os.path.join(args.out_dir, "kaggle_logs.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

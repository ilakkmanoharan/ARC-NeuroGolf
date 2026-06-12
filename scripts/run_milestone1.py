#!/usr/bin/env python3
"""Milestone 1: curated submission from audit + Kaggle submit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
from arc_genome.onnx.kaggle_score import kaggle_score_model
from scripts.audit_submission import audit

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
MILESTONE_DIR = "phases/milestone1"


def build_curated(
    source_dir: str,
    out_dir: str,
    min_kaggle_score: float = 1.0,
    require_arcgen: bool = True,
) -> dict:
    tasks = load_tasks_with_arcgen()
    os.makedirs(out_dir, exist_ok=True)

    included = []
    excluded = []

    for tn in sorted(tasks.keys()):
        src = os.path.join(source_dir, f"task{tn:03d}.onnx")
        if not os.path.exists(src):
            continue
        td = tasks[tn]["data"]
        if require_arcgen and not validate_full(src, td):
            excluded.append({"task": tn, "reason": "fails_arcgen"})
            continue
        ks = kaggle_score_model(src, td)
        if ks is None:
            excluded.append({"task": tn, "reason": "score_failed"})
            continue
        if ks["score"] < min_kaggle_score:
            excluded.append({"task": tn, "reason": f"low_score_{ks['score']:.2f}"})
            continue
        dst = os.path.join(out_dir, f"task{tn:03d}.onnx")
        shutil.copy2(src, dst)
        included.append({"task": tn, "kaggle_score": ks["score"], "kaggle_cost": ks["cost"]})

    zip_path = os.path.join(MILESTONE_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".onnx"):
                zf.write(os.path.join(out_dir, f), f)

    summary = {
        "included": len(included),
        "excluded": len(excluded),
        "kaggle_total_est": sum(x["kaggle_score"] for x in included),
        "min_kaggle_score": min_kaggle_score,
        "require_arcgen": require_arcgen,
    }
    with open(os.path.join(MILESTONE_DIR, "curated.json"), "w") as f:
        json.dump({"summary": summary, "included": included, "excluded": excluded}, f, indent=2)

    print(f"Curated: {len(included)} tasks, est. Kaggle score {summary['kaggle_total_est']:.1f}")
    print(f"Zip: {zip_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Milestone 1: measure, curate, submit")
    parser.add_argument("--source", default="phases/phase6/submission")
    parser.add_argument("--min_score", type=float, default=1.0)
    parser.add_argument("--no-arcgen", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--submit", action="store_true", default=True)
    parser.add_argument("--no-submit", dest="submit", action="store_false")
    args = parser.parse_args()

    os.makedirs(MILESTONE_DIR, exist_ok=True)
    os.makedirs(os.path.join(MILESTONE_DIR, "submission"), exist_ok=True)

    print("=== Step 1: Audit Phase 6 submission ===")
    audit(args.source, "data/all_tasks.json", os.path.join(MILESTONE_DIR, "audit.json"))

    if args.audit_only:
        return 0

    print("\n=== Step 2: Build curated submission ===")
    summary = build_curated(
        args.source,
        os.path.join(MILESTONE_DIR, "submission"),
        min_kaggle_score=args.min_score,
        require_arcgen=not args.no_arcgen,
    )

    if args.submit:
        msg = f"ARC-Genome M1: {summary['included']} tasks, est {summary['kaggle_total_est']:.0f}"
        cmd = [KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
               "-f", os.path.join(MILESTONE_DIR, "submission.zip"), "-m", msg]
        print("\n=== Step 3: Kaggle submit ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

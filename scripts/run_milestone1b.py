#!/usr/bin/env python3
"""Milestone 1b: re-solve with ARC-GEN fitting + validation, submit."""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import set_phase
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
MILESTONE_DIR = "phases/milestone1"
OUT = f"{MILESTONE_DIR}/submission_v2"


def main():
    set_phase(6)  # arcgen_validation + full pipeline
    os.makedirs(OUT, exist_ok=True)
    print("Re-solving 400 tasks with ARC-GEN validation...")
    results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
    zip_path = f"{MILESTONE_DIR}/submission_v2.zip"
    make_submission_zip(OUT, zip_path, "data/all_tasks.json")

    audit_path = f"{MILESTONE_DIR}/audit_v2.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT,
        "--out", audit_path,
    ], check=True)

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    results_doc = {
        "solved": len(results),
        "pass_all": summary["buckets"]["pass_all"],
        "kaggle_score_est": summary["pass_all_kaggle_total"],
        "zip": zip_path,
        "submitted": False,
    }
    with open(f"{MILESTONE_DIR}/results_v2.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    msg = f"ARC-Genome M1b: {len(results)} ARC-GEN verified, est {summary['pass_all_kaggle_total']:.0f}"
    submit_path = "submission.zip"
    import shutil
    shutil.copy2(zip_path, submit_path)
    print(f"\n=== Kaggle submit ({len(results)} tasks, est. {summary['pass_all_kaggle_total']:.1f}) ===")
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", submit_path, "-m", msg,
    ])
    if proc.returncode == 0:
        results_doc["submitted"] = True
        with open(f"{MILESTONE_DIR}/results_v2.json", "w") as f:
            json.dump(results_doc, f, indent=2)
        print("Submitted successfully.")
    else:
        print("Kaggle submit failed (often daily limit after many submissions).")
        print(f"Retry: {KAGGLE} competitions submit -c neurogolf-2026 -f {zip_path} -m \"{msg}\"")
        sys.exit(1)


if __name__ == "__main__":
    main()

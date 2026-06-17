#!/usr/bin/env python3
"""2026-06-17 submission: Phase 12 slim bounded flip."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import set_phase
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
DATE = "2026-06-17"
SUB_DIR = f"kaggle-submissions/{DATE}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"


def main():
    set_phase(12)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    print(f"Submission {DATE}: Phase 12 slim bounded flip...")
    results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
    make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")

    audit_path = f"{SUB_DIR}/audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT,
        "--out", audit_path,
        "--phase", "12",
    ], check=True)

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    from collections import Counter
    sc = Counter(r["solver"] for r in results.values())
    results_doc = {
        "date": DATE,
        "solved": len(results),
        "pass_all": summary["buckets"]["pass_all"],
        "kaggle_score_est": summary["pass_all_kaggle_total"],
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(sc.most_common()),
        "baseline_m6_kaggle": 834.44,
    }
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    n = summary["buckets"]["pass_all"]
    est = summary["pass_all_kaggle_total"]
    msg = f"ARC-Genome M7b: {n} verified, est {est:.0f}, slim bounded flip"
    print(f"\n=== Kaggle submit ({n} tasks, est. {est:.1f}) ===")
    shutil.copy2(ZIP_STORE, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        print("Submit failed")
        sys.exit(1)
    results_doc["submitted"] = True
    results_doc["message"] = msg
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)
    print("Submitted successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()

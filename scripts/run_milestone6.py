#!/usr/bin/env python3
"""Milestone 6: bounded Code World Models — dynamic bbox flip (Phase 11)."""

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
MILESTONE_DIR = "phases/milestone6"
OUT = f"{MILESTONE_DIR}/submission"
ZIP_STORE = f"{MILESTONE_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"


def main():
    set_phase(11)
    os.makedirs(MILESTONE_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    print("Milestone 6: bounded world models (dynamic bbox flip)...")
    results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
    make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")

    audit_path = f"{MILESTONE_DIR}/audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT,
        "--out", audit_path,
        "--phase", "11",
    ], check=True)

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    from collections import Counter
    sc = Counter(r["solver"] for r in results.values())
    results_doc = {
        "solved": len(results),
        "pass_all": summary["buckets"]["pass_all"],
        "kaggle_score_est": summary["pass_all_kaggle_total"],
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(sc.most_common()),
    }
    with open(f"{MILESTONE_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    n = summary["buckets"]["pass_all"]
    est = summary["pass_all_kaggle_total"]
    msg = f"ARC-Genome M6: {n} verified, est {est:.0f}, bounded CWM flip"
    print(f"\n=== Kaggle submit ({n} tasks, est. {est:.1f}) ===")
    shutil.copy2(ZIP_STORE, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        print("Submit failed — retry: python scripts/run_milestone6.py")
        sys.exit(1)
    results_doc["submitted"] = True
    results_doc["message"] = msg
    with open(f"{MILESTONE_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)
    print("Submitted successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()

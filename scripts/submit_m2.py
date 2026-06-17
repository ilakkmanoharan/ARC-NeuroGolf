#!/usr/bin/env python3
"""Submit Milestone 2 zip to Kaggle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
ZIP_PATH = os.path.join(ROOT, "phases/milestone2/submission_v2.zip")
SUBMIT_PATH = os.path.join(ROOT, "submission.zip")
AUDIT_PATH = os.path.join(ROOT, "phases/milestone2/audit.json")
RESULTS_PATH = os.path.join(ROOT, "phases/milestone2/results.json")


def main():
    if not os.path.isfile(ZIP_PATH):
        print(f"Missing {ZIP_PATH} — run scripts/run_milestone2.py first")
        sys.exit(1)
    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    n = audit["summary"]["buckets"]["pass_all"]
    est = audit["summary"]["pass_all_kaggle_total"]
    msg = f"ARC-Genome M2: {n} verified, est {est:.0f}, 100 arcgen"
    shutil.copy2(ZIP_PATH, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    results = {"submitted": True, "message": msg, "pass_all": n, "kaggle_score_est": est}
    if os.path.isfile(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = {**json.load(f), **results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("Submitted successfully.")


if __name__ == "__main__":
    main()

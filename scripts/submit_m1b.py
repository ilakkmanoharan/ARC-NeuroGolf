#!/usr/bin/env python3
"""Submit Milestone 1b zip to Kaggle (audit already done)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
ZIP_PATH = os.path.join(ROOT, "phases/milestone1/submission_v2.zip")
SUBMIT_PATH = os.path.join(ROOT, "submission.zip")  # Kaggle requires this filename
AUDIT_PATH = os.path.join(ROOT, "phases/milestone1/audit_v2.json")
RESULTS_PATH = os.path.join(ROOT, "phases/milestone1/results_v2.json")


def main():
    if not os.path.isfile(ZIP_PATH):
        print(f"Missing {ZIP_PATH} — run scripts/run_milestone1b.py first")
        sys.exit(1)

    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    n = audit["summary"]["buckets"]["pass_all"]
    est = audit["summary"]["pass_all_kaggle_total"]
    msg = f"ARC-Genome M1b: {n} ARC-GEN verified, est {est:.0f}"

    print(f"Submitting {ZIP_PATH}")
    print(f"  pass_all={n}, est_score={est:.1f}")
    print(f"  message: {msg}")

    import shutil
    shutil.copy2(ZIP_PATH, SUBMIT_PATH)

    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        print("Submit failed.")
        sys.exit(proc.returncode)

    results = {
        "solved": n,
        "pass_all": n,
        "kaggle_score_est": est,
        "zip": ZIP_PATH,
        "submitted": True,
        "message": msg,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("Submitted successfully.")


if __name__ == "__main__":
    main()

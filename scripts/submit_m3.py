#!/usr/bin/env python3
"""Submit Milestone 3 zip to Kaggle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
ZIP_PATH = os.path.join(ROOT, "phases/milestone3/submission_v2.zip")
SUBMIT_PATH = os.path.join(ROOT, "submission.zip")
AUDIT_PATH = os.path.join(ROOT, "phases/milestone3/audit.json")


def main():
    if not os.path.isfile(ZIP_PATH):
        sys.exit(f"Missing {ZIP_PATH}")
    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    n = audit["summary"]["buckets"]["pass_all"]
    est = audit["summary"]["pass_all_kaggle_total"]
    msg = f"ARC-Genome M3: {n} verified, est {est:.0f}, official score"
    shutil.copy2(ZIP_PATH, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

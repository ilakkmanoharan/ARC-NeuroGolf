#!/usr/bin/env python3
"""Submit submission.zip to Kaggle via CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


COMPETITION = "neurogolf-2026"


def main():
    parser = argparse.ArgumentParser(description="Submit to NeuroGolf 2026 on Kaggle")
    parser.add_argument("--zip_path", default="submission.zip")
    parser.add_argument("--message", default="ARC-Genome v0.1 baseline submission")
    args = parser.parse_args()

    if not os.path.exists(args.zip_path):
        print(f"Error: {args.zip_path} not found. Run scripts/solve_all.py first.")
        sys.exit(1)

    kaggle_bin = os.environ.get("KAGGLE_BIN", "kaggle")
    cmd = [
        kaggle_bin,
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        args.zip_path,
        "-m",
        args.message,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        alt = os.path.expanduser("~/Library/Python/3.9/bin/kaggle")
        if os.path.exists(alt):
            cmd[0] = alt
            print("Retrying with:", alt)
            result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a numbered phase: solve, zip, submit to Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import build_config, set_phase
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get(
    "KAGGLE_BIN",
    os.path.expanduser("~/Library/Python/3.9/bin/kaggle"),
)


def main():
    parser = argparse.ArgumentParser(description="Run ARC-Genome competition phase")
    parser.add_argument("phase", type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--data_file", default="data/all_tasks.json")
    parser.add_argument("--conv_budget", type=float, default=25.0)
    parser.add_argument("--submit", action="store_true", default=True)
    parser.add_argument("--no-submit", dest="submit", action="store_false")
    args = parser.parse_args()

    phase_dir = os.path.join("phases", f"phase{args.phase}")
    output_dir = os.path.join(phase_dir, "submission")
    zip_path = os.path.join(phase_dir, "submission.zip")
    os.makedirs(phase_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cfg = set_phase(args.phase)
    print(f"=== Phase {args.phase} ===")
    print(f"Config: {cfg}")
    print("=" * 70)

    t0 = time.time()
    results = solve_all(args.data_file, output_dir, args.conv_budget)
    make_submission_zip(output_dir, zip_path, args.data_file)

    summary = {
        "phase": args.phase,
        "solved": len(results),
        "total_tasks": 400,
        "elapsed_sec": round(time.time() - t0, 1),
        "solvers": {},
    }
    for r in results.values():
        s = r["solver"]
        summary["solvers"][s] = summary["solvers"].get(s, 0) + 1

    with open(os.path.join(phase_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to {phase_dir}/results.json")

    if args.submit:
        msg = f"ARC-Genome Phase {args.phase}: {len(results)}/400 solved"
        cmd = [KAGGLE, "competitions", "submit", "-c", "neurogolf-2026", "-f", zip_path, "-m", msg]
        print("Submitting:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

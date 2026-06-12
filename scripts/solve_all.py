#!/usr/bin/env python3
"""Solve all ARC tasks and create submission.zip."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.solve import make_submission_zip, solve_all


def main():
    parser = argparse.ArgumentParser(description="ARC-Genome NeuroGolf solver")
    parser.add_argument("--data_file", default="data/all_tasks.json")
    parser.add_argument("--output_dir", default="submission")
    parser.add_argument("--conv_budget", type=float, default=30.0)
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task numbers")
    parser.add_argument("--zip_path", default="submission.zip")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: {args.data_file} not found.")
        print("Download: curl -L https://huggingface.co/LuciferMrng/neurogolf-2026/resolve/main/all_tasks.json -o data/all_tasks.json")
        sys.exit(1)

    task_nums = [int(t) for t in args.tasks.split(",") if t.strip()] or None
    print(f"Conv budget: {args.conv_budget}s per task")
    print("=" * 70)

    solve_all(args.data_file, args.output_dir, args.conv_budget, task_nums)
    make_submission_zip(args.output_dir, args.zip_path, args.data_file)


if __name__ == "__main__":
    main()

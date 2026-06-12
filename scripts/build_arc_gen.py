#!/usr/bin/env python3
"""Verify ARC-GEN raw data and build compact arc_gen.json index."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    raw_dir = "data/arc_gen_raw"
    out_path = "data/arc_gen.json"
    tasks_path = "data/all_tasks.json"

    if not os.path.isdir(raw_dir):
        print(f"Missing {raw_dir}. Run: unzip data/ARC-GEN-100K.zip -d data/arc_gen_raw")
        sys.exit(1)

    with open(tasks_path) as f:
        all_tasks = json.load(f)

    arcgen: dict[str, list] = {}
    matched = 0
    for hex_id in sorted(all_tasks.keys()):
        path = os.path.join(raw_dir, f"{hex_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                arcgen[hex_id] = json.load(f)
            matched += 1

    with open(out_path, "w") as f:
        json.dump(arcgen, f)

    print(f"Built {out_path}: {matched}/400 tasks with ARC-GEN samples")
    if arcgen:
        sample = next(iter(arcgen.values()))
        print(f"Samples per task: {len(sample)}")


if __name__ == "__main__":
    main()

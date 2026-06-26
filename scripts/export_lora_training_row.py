#!/usr/bin/env python3
"""Append one training row for a LoRA adapter under training/lora-*/examples/."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lora_dataset import ADAPTER_GOALS, ADAPTER_NAMES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", choices=ADAPTER_NAMES.keys())
    parser.add_argument("--submission-dir", required=True, help="e.g. kaggle-submissions/2026-06-17/submission-4")
    parser.add_argument("--input-file", action="append", default=[], help="Input context file (repeatable)")
    parser.add_argument("--output-file", required=True, help="Model output file produced")
    args = parser.parse_args()

    sub = ROOT / args.submission_dir
    if not sub.is_dir():
        raise SystemExit(f"Missing {sub}")

    out_path = ROOT / args.output_file
    if not out_path.is_file():
        raise SystemExit(f"Missing output {out_path}")

    inputs: dict[str, str] = {"goal": ADAPTER_GOALS[args.adapter]}
    for rel in args.input_file:
        p = ROOT / rel
        if p.is_file():
            inputs[p.name] = p.read_text(encoding="utf-8")

    row = {
        "adapter": ADAPTER_NAMES[args.adapter],
        "submission_dir": args.submission_dir,
        "score_goal": ADAPTER_GOALS[args.adapter],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "output": out_path.read_text(encoding="utf-8"),
    }

    dest_dir = ROOT / "training" / f"lora-{args.adapter}" / "examples"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = dest_dir / f"{sub.name}_{stamp}.json"
    dest.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"Wrote training example {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

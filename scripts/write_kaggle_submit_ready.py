#!/usr/bin/env python3
"""Write kaggle_submit_ready.json after a solve when submit gates passed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create kaggle_submit_ready.json for GHA auto-submit")
    parser.add_argument("--submission-dir", required=True, help="e.g. kaggle-submissions/2026-06-26/submission-1")
    parser.add_argument("--message", default="", help="Kaggle submit message override")
    parser.add_argument("--force", action="store_true", help="Overwrite existing ready file")
    args = parser.parse_args()

    sub = Path(args.submission_dir)
    results_path = sub / "results.json"
    zip_path = sub / "submission_v2.zip"
    ready_path = sub / "kaggle_submit_ready.json"

    if ready_path.is_file() and not args.force:
        print(f"{ready_path} already exists")
        return 0

    if not zip_path.is_file():
        print(f"missing {zip_path}", file=sys.stderr)
        return 1

    results: dict = {}
    if results_path.is_file():
        results = json.loads(results_path.read_text())

    if results.get("submitted") is True:
        print("already submitted to Kaggle")
        return 0

    if not args.force and not results.get("ready_for_manual_kaggle_submit"):
        msg = results.get("message", "")
        print(f"submit gates not met (ready_for_manual_kaggle_submit=false): {msg or 'no message'}")
        return 0

    message = args.message.strip()
    if not message:
        message = results.get("message", "")
    if not message or message.startswith("No submit"):
        message = f"NeuroGolf: {results.get('pass_all', '?')} pass_all, est {results.get('kaggle_score_est', '?')}"

    payload = {"auto_submit": True, "message": message}
    ready_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {ready_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

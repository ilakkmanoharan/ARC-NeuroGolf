#!/usr/bin/env python3
"""Submit to Kaggle when gates pass. Used by run scripts and as local GHA fallback."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = os.environ.get(
    "KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle")
)


def submit_dir(sub_dir: Path, message: str | None = None, force: bool = False) -> int:
    sub_dir = sub_dir.resolve()
    results_path = sub_dir / "results.json"
    zip_path = sub_dir / "submission_v2.zip"
    ready_path = sub_dir / "kaggle_submit_ready.json"

    if not zip_path.is_file():
        print(f"missing {zip_path}", file=sys.stderr)
        return 1

    results: dict = {}
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    if results.get("submitted") is True and not force:
        print("already submitted")
        return 0

    if not force and not results.get("ready_for_manual_kaggle_submit"):
        if ready_path.is_file():
            cfg = json.loads(ready_path.read_text(encoding="utf-8"))
            if not cfg.get("auto_submit", False):
                print("auto_submit=false in ready file", file=sys.stderr)
                return 1
        else:
            print("submit gates not met", file=sys.stderr)
            return 1

    msg = message or results.get("message") or "NeuroGolf auto-submit"
    if msg.startswith("No submit"):
        print(f"refusing: {msg}", file=sys.stderr)
        return 1

    submit_zip = ROOT / "submission.zip"
    shutil.copy2(zip_path, submit_zip)
    proc = subprocess.run(
        [KAGGLE, "competitions", "submit", "-c", "neurogolf-2026", "-f", str(submit_zip), "-m", msg],
    )
    if proc.returncode != 0:
        return proc.returncode

    results["submitted"] = True
    results["message"] = msg
    results["submitted_at"] = datetime.now(timezone.utc).isoformat()
    results["submitted_by"] = os.environ.get("NEUROGOLF_SUBMIT_BY", "local:kaggle_auto_submit")
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"submitted {sub_dir.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit submission_v2.zip to Kaggle")
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--message", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return submit_dir(Path(args.submission_dir), args.message or None, args.force)


if __name__ == "__main__":
    raise SystemExit(main())

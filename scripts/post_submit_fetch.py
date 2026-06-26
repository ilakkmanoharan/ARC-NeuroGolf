#!/usr/bin/env python3
"""Post-submit: wait for Kaggle grade, fetch logs, update status files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wait_kaggle_complete import wait_for_complete


def _unzip_submission(sub_dir: str) -> str:
    onnx_dir = os.path.join(sub_dir, "submission")
    zip_path = os.path.join(sub_dir, "submission_v2.zip")
    if os.path.isdir(onnx_dir) and any(f.endswith(".onnx") for f in os.listdir(onnx_dir)):
        return onnx_dir
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"No submission ONNX or zip in {sub_dir}")
    os.makedirs(onnx_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", onnx_dir], check=True)
    return onnx_dir


def _logs_already_complete(sub_dir: str) -> bool:
    logs_json = os.path.join(sub_dir, "kaggle_logs", "kaggle_logs.json")
    results_path = os.path.join(sub_dir, "results.json")
    if not os.path.isfile(logs_json):
        return False
    try:
        with open(logs_json) as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        return False
    tasks = logs.get("tasks") if isinstance(logs, dict) else None
    if not tasks:
        return False
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        if results.get("kaggle_score_actual") is not None:
            return True
        if results.get("submitted") and len(tasks) >= 10:
            return True
    return len(tasks) >= 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--submission", type=int, required=True)
    parser.add_argument("--message", default="", help="Kaggle submission message substring")
    parser.add_argument("--initial-wait", type=int, default=600)
    parser.add_argument("--retry-wait", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=18)
    parser.add_argument("--skip-initial-wait", action="store_true")
    parser.add_argument("--download-official", action="store_true", default=True)
    args = parser.parse_args()

    sub_dir = f"kaggle-submissions/{args.date}/submission-{args.submission}"
    results_path = os.path.join(sub_dir, "results.json")

    if _logs_already_complete(sub_dir):
        print(f"Skip: kaggle_logs already complete for {sub_dir}")
        return

    message = args.message
    if not message and os.path.isfile(results_path):
        with open(results_path) as f:
            message = json.load(f).get("message", "")

    print(f"Waiting for Kaggle grade: {sub_dir}")
    if message:
        print(f"Matching message containing: {message[:80]!r}")

    grade = wait_for_complete(
        message_substr=message,
        initial_wait_s=args.initial_wait,
        retry_wait_s=args.retry_wait,
        max_retries=args.max_retries,
        skip_initial_wait=args.skip_initial_wait,
    )

    onnx_dir = _unzip_submission(sub_dir)
    logs_dir = os.path.join(sub_dir, "kaggle_logs")
    subprocess.run(
        [
            sys.executable,
            "scripts/fetch_kaggle_logs.py",
            "--submission_dir",
            onnx_dir,
            "--out_dir",
            logs_dir,
            *(["--download_official"] if args.download_official else []),
        ],
        check=True,
    )

    status_path = os.path.join(sub_dir, "kaggle_status.txt")
    proc = subprocess.run(
        [os.environ.get("KAGGLE_BIN", "kaggle"), "competitions", "submissions", "list",
         "-c", "neurogolf-2026", "--csv"],
        capture_output=True,
        text=True,
    )
    with open(status_path, "w") as f:
        f.write(f"# Kaggle status — submission-{args.submission}\n")
        f.write(f"# Graded at: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# Score: {grade.get('score')}\n")
        f.write(f"# Status: {grade.get('status')}\n\n")
        if proc.stdout:
            f.write(proc.stdout[:4000])

    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        results["kaggle_score_actual"] = grade.get("score")
        results["kaggle_graded_at"] = datetime.now(timezone.utc).isoformat()
        results["kaggle_status"] = grade.get("status")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Done: score={grade.get('score')} logs={logs_dir}")


if __name__ == "__main__":
    main()

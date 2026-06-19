#!/usr/bin/env python3
"""Wait for a Kaggle submission to complete and return the public score."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time

COMPETITION = "neurogolf-2026"
KAGGLE = os.environ.get("KAGGLE_BIN", "kaggle")


def _list_submissions() -> list[dict]:
    proc = subprocess.run(
        [KAGGLE, "competitions", "submissions", "list", "-c", COMPETITION, "--csv"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        raise RuntimeError("kaggle submissions list failed")
    reader = csv.DictReader(io.StringIO(proc.stdout))
    return list(reader)


def _parse_score(row: dict) -> float | None:
    for key in ("score", "publicScore", "publicScoreFullPrecision"):
        val = row.get(key, "").strip()
        if not val:
            continue
        try:
            return float(val)
        except ValueError:
            continue
    return None


def _is_complete(row: dict) -> bool:
    status = (row.get("status") or "").upper()
    if "COMPLETE" in status:
        return True
    if "ERROR" in status or "FAILED" in status:
        return False
    score = _parse_score(row)
    return score is not None and score > 0


def find_submission(
    *,
    message_substr: str = "",
    file_name: str = "",
    max_rows: int = 30,
) -> dict | None:
    rows = _list_submissions()
    for row in rows[:max_rows]:
        desc = row.get("description") or row.get("fileName") or ""
        fname = row.get("fileName") or ""
        if message_substr and message_substr not in desc:
            continue
        if file_name and file_name not in fname:
            continue
        return row
    if message_substr:
        for row in rows[:max_rows]:
            desc = row.get("description") or ""
            if message_substr in desc:
                return row
    return rows[0] if rows else None


def wait_for_complete(
    *,
    message_substr: str = "",
    initial_wait_s: int = 3600,
    retry_wait_s: int = 1800,
    max_retries: int = 4,
    skip_initial_wait: bool = False,
) -> dict:
    if not skip_initial_wait and initial_wait_s > 0:
        print(f"Waiting {initial_wait_s}s before first Kaggle status check...")
        time.sleep(initial_wait_s)

    for attempt in range(max_retries + 1):
        row = find_submission(message_substr=message_substr)
        if row is None:
            print("No submissions found on Kaggle")
        else:
            status = row.get("status", "")
            score = _parse_score(row)
            desc = row.get("description", row.get("fileName", ""))
            print(f"Check {attempt}: status={status!r} score={score} desc={desc[:80]!r}")
            if _is_complete(row):
                return {
                    "status": status,
                    "score": score,
                    "description": desc,
                    "date": row.get("date", ""),
                    "raw": row,
                }
        if attempt < max_retries:
            print(f"Not ready; sleeping {retry_wait_s}s ({attempt + 1}/{max_retries} retries left)")
            time.sleep(retry_wait_s)

    raise TimeoutError(
        f"Kaggle submission not COMPLETE after "
        f"{'0' if skip_initial_wait else str(initial_wait_s)}s initial + {max_retries}×{retry_wait_s}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Wait for Kaggle submission to complete")
    parser.add_argument("--message", default="", help="Substring to match submission description")
    parser.add_argument("--initial-wait", type=int, default=3600, help="Seconds before first check")
    parser.add_argument("--retry-wait", type=int, default=1800, help="Seconds between retries")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--skip-initial-wait", action="store_true")
    parser.add_argument("--out", default="", help="Write result JSON path")
    args = parser.parse_args()

    result = wait_for_complete(
        message_substr=args.message,
        initial_wait_s=args.initial_wait,
        retry_wait_s=args.retry_wait,
        max_retries=args.max_retries,
        skip_initial_wait=args.skip_initial_wait,
    )
    print(f"READY: score={result['score']} status={result['status']}")

    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

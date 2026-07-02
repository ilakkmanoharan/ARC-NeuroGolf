#!/usr/bin/env python3
"""Continuous submission lane — no calendar-day restriction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "kaggle-submissions"
SUB_RE = re.compile(r"^submission-(\d+)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def iter_submission_dirs() -> list[tuple[str, int, Path]]:
    rows: list[tuple[str, int, Path]] = []
    if not SUBMISSIONS.is_dir():
        return rows
    for date_dir in sorted(SUBMISSIONS.iterdir()):
        if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
            continue
        for sub_dir in sorted(date_dir.iterdir(), key=lambda p: int(SUB_RE.match(p.name).group(1)) if SUB_RE.match(p.name) else 0):
            m = SUB_RE.match(sub_dir.name)
            if m:
                rows.append((date_dir.name, int(m.group(1)), sub_dir))
    return rows


def _read_results(path: Path) -> dict:
    results = path / "results.json"
    if not results.is_file():
        return {}
    try:
        return json.loads(results.read_text())
    except json.JSONDecodeError:
        return {}


def latest_submitted() -> tuple[str, int, Path] | None:
    candidates: list[tuple[str, int, Path]] = []
    for date, num, path in iter_submission_dirs():
        data = _read_results(path)
        if data.get("submitted") is True:
            candidates.append((date, num, path))
    return max(candidates, key=lambda x: (x[0], x[1])) if candidates else None


def latest_folder() -> tuple[str, int, Path] | None:
    rows = iter_submission_dirs()
    return max(rows, key=lambda x: (x[0], x[1])) if rows else None


def active_date() -> str:
    """Date folder for new submissions — latest lane on disk, not calendar today."""
    latest = latest_folder()
    if latest:
        return latest[0]
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def find_run_script(date: str, num: int) -> Path | None:
    for name in (f"run_submission_{date}_s{num}.py", f"run_submission_{date}.py"):
        path = ROOT / "scripts" / name
        if path.is_file():
            return path
    return None


def work_dir() -> tuple[str, int, Path]:
    """Folder the agent should implement/solve next."""
    rows = iter_submission_dirs()
    if not rows:
        date = active_date()
        return date, 1, SUBMISSIONS / date / "submission-1"

    submitted = latest_submitted()
    sub_floor: tuple[str, int] | None = None
    if submitted:
        sub_floor = (submitted[0], submitted[1])

    for date, num, path in sorted(rows, key=lambda x: (x[0], x[1]), reverse=True):
        if (path / "superseded.md").is_file():
            continue
        if sub_floor and (date, num) <= sub_floor:
            continue
        if (path / "submission_v2.zip").is_file() and (path / "kaggle_submit_ready.json").is_file():
            continue
        if (path / "plan.md").is_file() or find_run_script(date, num):
            return date, num, path

    if submitted:
        s_date, s_num, _ = submitted
        return s_date, s_num + 1, SUBMISSIONS / s_date / f"submission-{s_num + 1}"

    date, num, _ = max(rows, key=lambda x: (x[0], x[1]))
    return date, num + 1, SUBMISSIONS / date / f"submission-{num + 1}"


def next_submission_dir() -> tuple[str, int, Path]:
    """Folder to plan after diagnosing latest submitted."""
    work = work_dir()
    submitted = latest_submitted()
    if not submitted:
        return work
    s_date, s_num, _ = submitted
    w_date, w_num, _ = work
    if (w_date, w_num) > (s_date, s_num):
        return work
    return s_date, s_num + 1, SUBMISSIONS / s_date / f"submission-{s_num + 1}"


def lane_state() -> dict:
    sub = latest_submitted()
    work = work_dir()
    nxt = next_submission_dir()
    return {
        "active_date": active_date(),
        "latest_submitted": (
            {"date": sub[0], "submission": sub[1], "path": str(sub[2].relative_to(ROOT))} if sub else None
        ),
        "work_dir": {"date": work[0], "submission": work[1], "path": str(work[2].relative_to(ROOT))},
        "next_submission": {"date": nxt[0], "submission": nxt[1], "path": str(nxt[2].relative_to(ROOT))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous NeuroGolf submission lane")
    parser.add_argument("--export-latest-submitted", action="store_true", help="Print DATE= and SUB_NUM= for bash")
    parser.add_argument("--export-work-dir", action="store_true", help="Print work dir DATE= and SUB_NUM=")
    parser.add_argument("--json", action="store_true", help="Print lane state as JSON")
    args = parser.parse_args()

    if args.export_latest_submitted:
        sub = latest_submitted()
        if not sub:
            print("::error::No submitted submission found on disk", file=sys.stderr)
            return 1
        date, num, _ = sub
        print(f"DATE={date}")
        print(f"SUB_NUM={num}")
        return 0

    if args.export_work_dir:
        date, num, _ = work_dir()
        print(f"DATE={date}")
        print(f"SUB_NUM={num}")
        return 0

    if args.json:
        print(json.dumps(lane_state(), indent=2))
        return 0

    print(json.dumps(lane_state(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

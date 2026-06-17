#!/usr/bin/env python3
"""Append a row to kaggle-submissions/all-submissions.md."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone

REGISTRY = "kaggle-submissions/all-submissions.md"

HEADER = """# All Submissions

Master log of [NeuroGolf 2026](https://kaggle.com/competitions/neurogolf-2026) submissions.
Rows are appended by `scripts/append_submission_registry.py` (CI or local).

| Date | Time (UTC) | # | Folder | Notebook | pass_all | Est score | Submitted | Description |
|---|---:|---:|---|---|---:|---:|---|---|
"""


def _fmt_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "—"


def _parse_folder(folder: str) -> tuple[str, int | None]:
    m = re.search(r"kaggle-submissions/(\d{4}-\d{2}-\d{2})/submission-(\d+)", folder)
    if m:
        return m.group(1), int(m.group(2))
    return "", None


def _already_registered(date: str, sub_num) -> bool:
    if not os.path.exists(REGISTRY) or not date or sub_num is None:
        return False
    needle = f"| {date} |"
    with open(REGISTRY) as f:
        for line in f:
            if not line.startswith("|") or line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 5:
                continue
            if parts[1] == date and parts[3] == str(sub_num):
                return True
    return False


def append_row(
    folder: str,
    *,
    description: str = "",
    notebook: str = "",
    results_path: str = "",
    force: bool = False,
) -> str:
    folder = folder.rstrip("/")
    results_path = results_path or os.path.join(folder, "results.json")

    data: dict = {}
    if os.path.isfile(results_path):
        with open(results_path) as f:
            data = json.load(f)

    date = data.get("date") or _parse_folder(folder)[0]
    sub_num = data.get("submission", _parse_folder(folder)[1])
    pass_all = data.get("pass_all", "—")
    est = data.get("kaggle_score_est")
    submitted = bool(data.get("submitted", False))
    message = description or data.get("message", "")

    nb_path = notebook or os.path.join(folder, "kaggle_notebook.md")
    notebook_cell = f"`{nb_path}`" if os.path.isfile(nb_path) else "—"

    if not force and _already_registered(date, sub_num):
        print(f"Skip: submission-{sub_num} on {date} already in {REGISTRY}")
        return REGISTRY

    now = datetime.now(timezone.utc)
    row = (
        f"| {date or '—'} "
        f"| {now.strftime('%H:%M')} "
        f"| {sub_num if sub_num is not None else '—'} "
        f"| `{folder}/` "
        f"| {notebook_cell} "
        f"| {pass_all} "
        f"| {_fmt_score(est)} "
        f"| {'yes' if submitted else 'no'} "
        f"| {message} |"
    )

    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    if not os.path.exists(REGISTRY):
        with open(REGISTRY, "w") as f:
            f.write(HEADER)
            f.write(row + "\n")
    else:
        with open(REGISTRY, "a") as f:
            f.write(row + "\n")

    print(f"Appended submission-{sub_num} ({date}) to {REGISTRY}")
    return REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Append submission row to all-submissions.md")
    parser.add_argument("--folder", required=True, help="e.g. kaggle-submissions/2026-06-17/submission-4")
    parser.add_argument("--description", default="", help="Override description column")
    parser.add_argument("--notebook", default="", help="Path to kaggle_notebook.md")
    parser.add_argument("--results", default="", help="Override results.json path")
    parser.add_argument("--force", action="store_true", help="Append even if date+# already present")
    args = parser.parse_args()
    append_row(
        args.folder,
        description=args.description,
        notebook=args.notebook,
        results_path=args.results,
        force=args.force,
    )


if __name__ == "__main__":
    main()

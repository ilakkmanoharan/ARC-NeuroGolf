#!/usr/bin/env python3
"""Queue the next NeuroGolf submission without Cursor API (continual harness fallback)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIGGERS = ROOT / ".github" / "triggers"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commit",
        action="store_true",
        help="git add/commit/push trigger JSON (for GHA post-submit step)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print trigger path only")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from submission_lane import find_run_script, latest_submitted, work_dir

    date, num, path = work_dir()
    script = find_run_script(date, num)
    if script is None:
        print(f"No run script for {date} submission-{num} — agent must implement first", file=sys.stderr)
        return 0

    if (path / "submission_v2.zip").is_file() and (path / "kaggle_submit_ready.json").is_file():
        print(f"submission-{num} already has zip + submit ready — skip queue", file=sys.stderr)
        return 0

    sub = latest_submitted()
    prior_n = sub[1] if sub else max(0, num - 1)
    trigger_name = f"submission-{date}-s{num}.json"
    trigger_path = TRIGGERS / trigger_name
    cfg = {
        "date": date,
        "submission_number": str(num),
        "run_script": script.name,
        "description": f"M14 Phase 24 continual harness — after submission-{prior_n}",
        "fetch_prev_logs": True,
        "auto_commit": True,
        "chain_post_submit": True,
    }

    if trigger_path.is_file():
        existing = json.loads(trigger_path.read_text())
        if existing.get("run_script") == cfg["run_script"]:
            print(f"Trigger already exists: {trigger_path.relative_to(ROOT)}")
            return 0

    print(f"Queue {date} submission-{num} via {script.name}")
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0

    TRIGGERS.mkdir(parents=True, exist_ok=True)
    trigger_path.write_text(json.dumps(cfg, indent=2) + "\n")

    if not args.commit:
        print(f"Wrote {trigger_path.relative_to(ROOT)} (not committed)")
        return 0

    subprocess.run(["git", "add", str(trigger_path)], cwd=ROOT, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if diff.returncode == 0:
        print("Nothing new to commit")
        return 0

    msg = f"ci: queue submission-{num} ({date}) continual harness"
    subprocess.run(
        ["git", "commit", "-m", msg, "-m", f"Auto-queued {script.name} after post-submit."],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=ROOT, check=True)
    print("Pushed trigger — neurogolf-submission-trigger.yml will dispatch solve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

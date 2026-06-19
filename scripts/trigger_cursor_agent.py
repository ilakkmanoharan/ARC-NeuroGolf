#!/usr/bin/env python3
"""Trigger Cursor Cloud Agent from CI (analysis or implement next submission)."""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Target submission date YYYY-MM-DD")
    parser.add_argument("--submission", type=int, required=True, help="Target submission number")
    parser.add_argument(
        "--mode",
        choices=["analysis", "implement"],
        default="implement",
        help="analysis: grade prior submission; implement: plan+code next submission",
    )
    parser.add_argument(
        "--prior-date",
        default="",
        help="Date folder for submission to analyze (default: same as --date, submission-1)",
    )
    parser.add_argument("--prior-submission", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        print("CURSOR_API_KEY not set — skip Cursor agent step", file=sys.stderr)
        sys.exit(0)

    try:
        from cursor_sdk import Agent, AgentOptions, CloudAgentOptions
    except ImportError:
        print("cursor-sdk not installed — pip install cursor-sdk", file=sys.stderr)
        sys.exit(1)

    repo = os.environ.get("GITHUB_REPOSITORY", "ilakkmanoharan/ARC-NeuroGolf")
    n = args.submission
    date = args.date

    if args.mode == "analysis":
        prior_n = args.prior_submission or max(1, n - 1)
        prior_date = args.prior_date or date
        prompt = f"""Analyze NeuroGolf submission-{prior_n} under kaggle-submissions/{prior_date}/submission-{prior_n}/.

Read kaggle_logs/kaggle_logs.json, results.json, audit.json, and prior submission docs in that day folder.

Write under kaggle-submissions/{prior_date}/submission-{prior_n}/:
- analysis.md (score delta, solver mix, regressions, audit vs Kaggle ratio)
- theory.md (BLF / Code World Models framing)
- learnings.md (actionable rules for next submission)

Match the style of existing submission-2 analysis on 2026-06-17. Open a PR only — do not change arc_genome solver code."""
    else:
        prior_n = args.prior_submission or max(1, n - 1)
        prior_date = args.prior_date or date
        if prior_date == date and prior_n >= n:
            prior_n = n - 1
        if prior_n < 1:
            prior_date = "2026-06-17"
            prior_n = 4
        prompt = f"""Implement NeuroGolf submission-{n} for date {date} (M10+).

Study kaggle-submissions/{prior_date}/submission-{prior_n}/ analysis, theory, learnings, and kaggle_logs.

Create kaggle-submissions/{date}/submission-{n}/:
- plan.md, strategy.md, theory.md

Implement minimal solver improvements in arc_genome/ per the plan (object programs / spatial gather / cheaper bounded compile).
Add scripts/run_submission_{date}_s{n}.py — follow run_submission_2026-06-17_s4.py:
  seed from prior submission zip, phase config, solve_all, audit, conditional Kaggle submit.

Rules: 100 ARC-GEN gate; no train_only tasks; gate submit on pass_all increase or est score +1.
Open a PR only. Do not push to main."""

    print(f"Launching Cursor Cloud Agent ({args.mode}) for {date} submission-{n}...")
    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=api_key,
            model="composer-2.5",
            cloud=CloudAgentOptions(
                repos=[repo],
                auto_create_pr=True,
                skip_reviewer_request=True,
            ),
        ),
    )

    if result.status == "error":
        print(f"Agent run failed: {result}", file=sys.stderr)
        sys.exit(2)
    print(f"Agent finished: status={result.status}")
    agent_id = getattr(result, "agent_id", None) or getattr(result, "id", None)
    if agent_id:
        print(f"Agent ID: {agent_id}")


if __name__ "__main__":
    main()

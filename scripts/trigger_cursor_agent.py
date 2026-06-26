#!/usr/bin/env python3
"""Trigger Cursor Cloud Agent from CI (Agent 1 full loop or legacy modes)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT1_INSTRUCTIONS = ROOT / "docs" / "agent1" / "instructions.md"


def _load_agent1_prompt() -> str:
    if not AGENT1_INSTRUCTIONS.is_file():
        print(f"Missing {AGENT1_INSTRUCTIONS}", file=sys.stderr)
        sys.exit(1)
    return AGENT1_INSTRUCTIONS.read_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="", help="Target submission date YYYY-MM-DD (legacy modes)")
    parser.add_argument("--submission", type=int, default=0, help="Target submission number (legacy modes)")
    parser.add_argument(
        "--mode",
        choices=["agent1", "analysis", "implement"],
        default="agent1",
        help="agent1: full loop per docs/agent1/instructions.md; legacy modes for one-off tasks",
    )
    parser.add_argument(
        "--prior-date",
        default="",
        help="Date folder for submission to analyze (legacy implement/analysis)",
    )
    parser.add_argument("--prior-submission", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        print("CURSOR_API_KEY not set — skip Cursor agent step", file=sys.stderr)
        sys.exit(0)

    try:
        from cursor_sdk import Agent, AgentOptions, CloudAgentOptions, CloudRepository, CursorAgentError
    except ImportError:
        print("cursor-sdk not installed — pip install cursor-sdk", file=sys.stderr)
        sys.exit(1)

    repo_slug = os.environ.get("GITHUB_REPOSITORY", "ilakkmanoharan/ARC-NeuroGolf")
    repo_url = f"https://github.com/{repo_slug}"

    if args.mode == "agent1":
        prompt = _load_agent1_prompt()
        print("Launching Cursor Cloud Agent (Agent 1 full loop)...")
    elif args.mode == "analysis":
        prior_n = args.prior_submission or max(1, args.submission - 1)
        prior_date = args.prior_date or args.date
        prompt = f"""Analyze NeuroGolf submission-{prior_n} under kaggle-submissions/{prior_date}/submission-{prior_n}/.

Read kaggle_logs/kaggle_logs.json, results.json, audit.json, and prior submission docs in that day folder.

Write under kaggle-submissions/{prior_date}/submission-{prior_n}/:
- analysis.md (score delta, solver mix, regressions, audit vs Kaggle ratio)
- theory.md (BLF / Code World Models framing)
- learnings.md (actionable rules for next submission)

Match the style of existing submission-2 analysis on 2026-06-17. Push directly to main — do not open a PR."""
        print(f"Launching Cursor Cloud Agent (analysis) for {prior_date} submission-{prior_n}...")
    else:
        n = args.submission
        date = args.date
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

Implement minimal solver improvements in arc_genome/ per the plan.
Add scripts/run_submission_{date}_s{n}.py — follow run_submission_2026-06-17_s4.py:
  seed from prior submission zip, phase config, solve_all, audit.
Set NEUROGOLF_SKIP_KAGGLE_SUBMIT=1 during solve; write kaggle_submit_ready.json after solve.

Rules: 100 ARC-GEN gate; no train_only tasks; gate submit on pass_all increase or est score +1.
Push directly to main — do not open a PR."""
        print(f"Launching Cursor Cloud Agent (implement) for {date} submission-{n}...")

    # Fire-and-forget: solve takes 75–90 min; GHA must not block on run.wait().
    agent = Agent.create(
        AgentOptions(
            api_key=api_key,
            model="composer-2.5",
            cloud=CloudAgentOptions(
                repos=[CloudRepository(url=repo_url, starting_ref="main")],
                auto_create_pr=False,
                skip_reviewer_request=True,
            ),
        ),
    )
    try:
        run = agent.send(prompt)
        print("Launched Cursor Cloud Agent (fire-and-forget)")
        print(f"Agent ID: {agent.agent_id}")
        run_id = getattr(run, "run_id", None)
        if run_id:
            print(f"Run ID: {run_id}")
        print("Solve runs ~75–90 min on Cursor cloud. Watch: Cursor → Agents (filter Source → SDK)")
    except CursorAgentError as err:
        print(f"Agent startup failed: {err.message}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            agent.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

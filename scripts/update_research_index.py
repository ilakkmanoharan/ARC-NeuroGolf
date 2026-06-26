#!/usr/bin/env python3
"""Rebuild kaggle-submissions/research/index.html from analysis/theory docs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "kaggle-submissions"
OUT = SUBMISSIONS / "research" / "index.html"


def _parse_submission_dir(path: Path) -> tuple[str, int] | None:
    m = re.match(r"submission-(\d+)$", path.name)
    if not m or not re.match(r"\d{4}-\d{2}-\d{2}$", path.parent.name):
        return None
    return path.parent.name, int(m.group(1))


def _score_from_results(sub_dir: Path) -> str:
    results = sub_dir / "results.json"
    if not results.is_file():
        return "—"
    try:
        data = json.loads(results.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "—"
    actual = data.get("kaggle_score_actual")
    if actual is not None:
        return f"{float(actual):.2f}"
    est = data.get("kaggle_score_est")
    if est is not None:
        return f"~{float(est):.0f} (est)"
    return "—"


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    if not SUBMISSIONS.is_dir():
        return rows
    for date_dir in sorted(SUBMISSIONS.iterdir(), reverse=True):
        if not date_dir.is_dir() or not re.match(r"\d{4}-\d{2}-\d{2}$", date_dir.name):
            continue
        for sub_dir in sorted(
            (p for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("submission-")),
            key=lambda p: int(p.name.split("-")[1]),
            reverse=True,
        ):
            parsed = _parse_submission_dir(sub_dir)
            if not parsed:
                continue
            date, num = parsed
            if not (sub_dir / "analysis.md").is_file() and not (sub_dir / "theory.md").is_file():
                continue
            rel = sub_dir.relative_to(SUBMISSIONS)
            rows.append(
                {
                    "date": date,
                    "num": num,
                    "score": _score_from_results(sub_dir),
                    "analysis": f"../{rel}/analysis.md" if (sub_dir / "analysis.md").is_file() else "",
                    "theory": f"../{rel}/theory.md" if (sub_dir / "theory.md").is_file() else "",
                    "plan": f"../{rel}/plan.md" if (sub_dir / "plan.md").is_file() else "",
                    "page": f"../{rel}/page.html" if (sub_dir / "page.html").is_file() else "",
                }
            )
    return rows


def render(rows: list[dict]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body_rows = []
    for r in rows:
        links = []
        if r["analysis"]:
            links.append(f'<a href="{r["analysis"]}">analysis</a>')
        if r["theory"]:
            links.append(f'<a href="{r["theory"]}">theory</a>')
        if r["plan"]:
            links.append(f'<a href="{r["plan"]}">plan</a>')
        if r["page"]:
            links.append(f'<a href="{r["page"]}">report</a>')
        link_cell = " · ".join(links) if links else "—"
        body_rows.append(
            f"""        <tr>
          <td>{r['date']}</td>
          <td>{r['num']}</td>
          <td>{r['score']}</td>
          <td>{link_cell}</td>
        </tr>"""
        )
    if not body_rows:
        body_rows.append(
            '        <tr><td colspan="4">No research docs yet — run Agent 1 step 2.</td></tr>'
        )
    tbody = "\n".join(body_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NeuroGolf Research</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #ffffff;
      --text: #172033;
      --muted: #5b6475;
      --border: #d9dee8;
      --soft: #f6f8fb;
      --accent: #17345f;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    main {{
      width: min(960px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 28px;
      background: linear-gradient(180deg, #f9fbff 0%, #ffffff 100%);
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0 0 8px; color: var(--accent); font-size: 2.2rem; }}
    p {{ margin: 0; color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--border);
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
    }}
    th {{ background: var(--soft); color: var(--accent); }}
    a {{ color: #154fa3; }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 0.9rem; }}
    code {{ background: var(--soft); padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>NeuroGolf Research</h1>
      <p>Analysis, theory, and plans per submission. Updated by Agent 1 (LoRA <code>NeuroGolf-Diagnose</code> / <code>NeuroGolf-Strategize</code>).</p>
      <p style="margin-top:12px"><a href="../gallery/index.html">Submission gallery</a></p>
    </header>
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>#</th>
          <th>Score</th>
          <th>Docs</th>
        </tr>
      </thead>
      <tbody>
{tbody}
      </tbody>
    </table>
    <footer>Generated {generated} by <code>scripts/update_research_index.py</code></footer>
  </main>
</body>
</html>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    OUT.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {OUT} ({len(rows)} submissions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

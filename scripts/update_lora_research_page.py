#!/usr/bin/env python3
"""Build LoRA + score progress research page with charts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lora_dataset import ADAPTER_NAMES, discover_submissions, score_delta_label  # noqa: E402

OUT_DIR = ROOT / "kaggle-submissions" / "research" / "lora"
STATS_JSON = OUT_DIR / "stats.json"
INDEX_HTML = OUT_DIR / "index.html"
PORTFOLIO_STATS = Path.home() / "Projects" / "ilakk-manoharan" / "content" / "neurogolf" / "lora-stats.json"


def _count_examples(adapter: str) -> int:
    d = ROOT / "training" / f"lora-{adapter}" / "examples"
    return len(list(d.glob("*.json"))) if d.is_dir() else 0


def _checkpoint_exists(adapter: str) -> bool:
    return (ROOT / "training" / f"lora-{adapter}" / "checkpoints" / "adapter_config.json").is_file()


def build_stats() -> dict:
    records = discover_submissions()
    scored = [r for r in records if r.kaggle_actual is not None]
    timeline = [
        {
            "label": f"{r.date} s{r.num}",
            "date": r.date,
            "submission": r.num,
            "kaggle": r.kaggle_actual,
            "pass_all": r.pass_all,
            "est": r.kaggle_est,
        }
        for r in scored
    ]
    outcomes = {"score_up": 0, "score_flat": 0, "score_down": 0, "baseline": 0}
    for i, r in enumerate(scored):
        prev = scored[i - 1] if i > 0 else None
        outcomes[score_delta_label(prev, r)] = outcomes.get(score_delta_label(prev, r), 0) + 1

    adapters = {}
    for name in ("diagnose", "strategize", "implement"):
        adapters[name] = {
            "display": ADAPTER_NAMES[name],
            "examples": _count_examples(name),
            "mlx_train_rows": sum(
                1 for _ in (ROOT / "training" / f"lora-{name}" / "mlx" / "train.jsonl").open()
            )
            if (ROOT / "training" / f"lora-{name}" / "mlx" / "train.jsonl").is_file()
            else 0,
            "checkpoint": _checkpoint_exists(name),
            "base_model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        }

    best = max(scored, key=lambda r: r.kaggle_actual or 0) if scored else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "competition": "neurogolf-2026",
        "goal": "Increase Kaggle public score via research agents + LoRA-guided plans",
        "best_kaggle": best.kaggle_actual if best else None,
        "best_pass_all": best.pass_all if best else None,
        "best_label": f"{best.date} submission-{best.num}" if best else None,
        "timeline": timeline,
        "outcomes": outcomes,
        "adapters": adapters,
        "note_arcgen": (
            "ARC-GEN JSON grids train ONNX solvers (arc_genome/), not LoRA text. "
            "LoRA synthetic rows are built from analysis.md, plan.md, and run scripts."
        ),
    }


def render_html(stats: dict) -> str:
    data_json = json.dumps(stats)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NeuroGolf LoRA Research</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{ color-scheme: light; --bg:#fff; --text:#172033; --muted:#5b6475; --border:#d9dee8; --accent:#17345f; }}
    body {{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); }}
    main {{ width:min(1100px,calc(100% - 32px)); margin:0 auto; padding:32px 0 48px; }}
    header {{ border:1px solid var(--border); border-radius:14px; padding:28px; margin-bottom:24px; background:linear-gradient(180deg,#f9fbff,#fff); }}
    h1 {{ margin:0 0 8px; color:var(--accent); font-size:2rem; }}
    .muted {{ color:var(--muted); }}
    .grid {{ display:grid; gap:20px; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); margin-bottom:24px; }}
    .card {{ border:1px solid var(--border); border-radius:12px; padding:16px; background:#fff; }}
    .stat {{ font-size:1.8rem; font-weight:700; color:var(--accent); }}
    canvas {{ max-height:320px; }}
    table {{ width:100%; border-collapse:collapse; border:1px solid var(--border); }}
    th,td {{ border:1px solid var(--border); padding:8px 10px; text-align:left; }}
    th {{ background:#f6f8fb; }}
    a {{ color:#154fa3; }}
    footer {{ margin-top:28px; color:var(--muted); font-size:.9rem; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>NeuroGolf LoRA Research</h1>
      <p class="muted">Goal: <strong>increase Kaggle score</strong> — adapters learn from scored submissions which levers moved the needle.</p>
      <p class="muted"><a href="../index.html">Submission research</a> · <a href="https://github.com/ilakkmanoharan/ARC-NeuroGolf">GitHub</a></p>
    </header>
    <div class="grid" id="kpis"></div>
    <div class="grid">
      <div class="card"><h2>Kaggle score timeline</h2><canvas id="chartScore"></canvas></div>
      <div class="card"><h2>pass_all tasks</h2><canvas id="chartPass"></canvas></div>
      <div class="card"><h2>LoRA training rows</h2><canvas id="chartLora"></canvas></div>
      <div class="card"><h2>Submission outcomes</h2><canvas id="chartOutcomes"></canvas></div>
    </div>
    <div class="card">
      <h2>Adapters</h2>
      <table id="adapterTable"><thead><tr><th>Adapter</th><th>Role</th><th>Examples</th><th>MLX rows</th><th>Checkpoint</th><th>Base model</th></tr></thead><tbody></tbody></table>
      <p class="muted" style="margin-top:12px" id="arcgenNote"></p>
    </div>
    <footer>Generated by <code>scripts/update_lora_research_page.py</code></footer>
  </main>
  <script>
    const STATS = {data_json};
    const kpi = document.getElementById('kpis');
    kpi.innerHTML = `
      <div class="card"><div class="muted">Best Kaggle</div><div class="stat">${{STATS.best_kaggle ?? '—'}}</div><div class="muted">${{STATS.best_label ?? ''}}</div></div>
      <div class="card"><div class="muted">Best pass_all</div><div class="stat">${{STATS.best_pass_all ?? '—'}}</div></div>
      <div class="card"><div class="muted">Scored submissions</div><div class="stat">${{STATS.timeline.length}}</div></div>`;
    document.getElementById('arcgenNote').textContent = STATS.note_arcgen;
    const labels = STATS.timeline.map(t => t.label);
    new Chart(document.getElementById('chartScore'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Kaggle', data: STATS.timeline.map(t=>t.kaggle), borderColor:'#17345f', tension:0.2 }}] }},
      options: {{ scales: {{ y: {{ beginAtZero: false }} }} }}
    }});
    new Chart(document.getElementById('chartPass'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 'pass_all', data: STATS.timeline.map(t=>t.pass_all), backgroundColor:'#4a7ab8' }}] }}
    }});
    const ad = STATS.adapters;
    new Chart(document.getElementById('chartLora'), {{
      type: 'bar',
      data: {{
        labels: Object.keys(ad).map(k => ad[k].display),
        datasets: [{{ label: 'examples', data: Object.keys(ad).map(k => ad[k].examples), backgroundColor:'#2d8a6e' }}]
      }}
    }});
    const oc = STATS.outcomes;
    new Chart(document.getElementById('chartOutcomes'), {{
      type: 'doughnut',
      data: {{
        labels: Object.keys(oc),
        datasets: [{{ data: Object.values(oc), backgroundColor:['#2d8a6e','#c9a227','#c44','#888'] }}]
      }}
    }});
    const tbody = document.querySelector('#adapterTable tbody');
    for (const k of Object.keys(ad)) {{
      const row = ad[k];
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{k}}</td><td>${{row.display}}</td><td>${{row.examples}}</td><td>${{row.mlx_train_rows}}</td><td>${{row.checkpoint ? 'yes' : 'no'}}</td><td>${{row.base_model}}</td>`;
      tbody.appendChild(tr);
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    stats = build_stats()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    INDEX_HTML.write_text(render_html(stats), encoding="utf-8")
    if PORTFOLIO_STATS.parent.is_dir():
        PORTFOLIO_STATS.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote portfolio stats {PORTFOLIO_STATS}")
    print(f"Wrote {INDEX_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

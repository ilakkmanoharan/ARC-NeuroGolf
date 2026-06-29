#!/usr/bin/env python3
"""Bootstrap LoRA training examples from all kaggle-submissions on disk."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_dataset import (
    ADAPTER_GOALS,
    ADAPTER_NAMES,
    ROOT,
    discover_submissions,
    export_mlx_jsonl,
    format_instruction,
    score_delta_label,
    summarize_arcgen_role,
    summarize_kaggle_logs,
)


def _read(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _write_example(adapter: str, sub_dir: Path, row: dict) -> Path:
    dest = ROOT / "training" / f"lora-{adapter}" / "examples"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = dest / f"{sub_dir.name}_{stamp}.json"
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return out


def _prior_submission(records, cur):
    prior = [r for r in records if (r.date, r.num) < (cur.date, cur.num)]
    return prior[-1] if prior else None


def bootstrap_diagnose(records) -> list[str]:
    texts: list[str] = []
    for rec in records:
        sub = rec.path
        analysis = sub / "analysis.md"
        if not analysis.is_file():
            continue
        prev = _prior_submission(records, rec)
        label = score_delta_label(prev, rec)
        prompt_parts = [
            f"You are {ADAPTER_NAMES['diagnose']}. {ADAPTER_GOALS['diagnose']}",
            f"Analyze NeuroGolf submission-{rec.num} ({rec.date}).",
            f"score_outcome={label}",
        ]
        if prev:
            prompt_parts.append(
                f"prior_submission={prev.date}/submission-{prev.num} kaggle={prev.kaggle_actual}"
            )
        if rec.kaggle_actual is not None:
            prompt_parts.append(f"kaggle_actual={rec.kaggle_actual} pass_all={rec.pass_all}")
        logs = summarize_kaggle_logs(sub / "kaggle_logs" / "kaggle_logs.json")
        if logs:
            prompt_parts.append(logs)
        prompt_parts.append(summarize_arcgen_role(sub))
        results = _read(sub / "results.json")
        if results:
            prompt_parts.append(f"## results.json\n{results[:4000]}")
        prompt = "\n\n".join(prompt_parts)
        response = _read(analysis)
        row = {
            "adapter": ADAPTER_NAMES["diagnose"],
            "submission_dir": rec.rel(),
            "score_outcome": label,
            "inputs": {"prompt": prompt},
            "output": response,
        }
        _write_example("diagnose", sub, row)
        texts.append(format_instruction(prompt, response))
        for name in ("theory.md", "learnings.md"):
            extra = _read(sub / name)
            if extra:
                texts.append(format_instruction(prompt, extra))
    return texts


def bootstrap_strategize(records) -> list[str]:
    texts: list[str] = []
    for rec in records:
        sub = rec.path
        plan = sub / "plan.md"
        if not plan.is_file():
            continue
        prev = _prior_submission(records, rec)
        label = score_delta_label(prev, rec)
        prompt_parts = [
            f"You are {ADAPTER_NAMES['strategize']}. {ADAPTER_GOALS['strategize']}",
            f"Plan submission-{rec.num} ({rec.date}).",
            f"prior score_outcome={label}",
        ]
        if prev and prev.kaggle_actual is not None:
            prompt_parts.append(f"prior_kaggle={prev.kaggle_actual} prior_pass_all={prev.pass_all}")
        if prev:
            analysis_src = prev.path / "analysis.md"
            if analysis_src.is_file():
                prompt_parts.append(f"## prior analysis\n{_read(analysis_src)[:6000]}")
        prompt_parts.append(summarize_arcgen_role(sub))
        prompt = "\n\n".join(prompt_parts)
        response = _read(plan)
        strategy = _read(sub / "strategy.md")
        if strategy:
            response = response + "\n\n---\n\n# Strategy\n\n" + strategy
        june = ROOT / "strategy" / "June-29-2026" / "strategy.md"
        if june.is_file() and rec.date >= "2026-06-26":
            response = response + "\n\n---\n\n# Session strategy (June-29)\n\n" + _read(june)[:8000]
        row = {
            "adapter": ADAPTER_NAMES["strategize"],
            "submission_dir": rec.rel(),
            "score_outcome": label,
            "inputs": {"prompt": prompt},
            "output": response,
        }
        _write_example("strategize", sub, row)
        texts.append(format_instruction(prompt, response))
    return texts


def bootstrap_implement(records) -> list[str]:
    texts: list[str] = []
    for rec in records:
        sub = rec.path
        plan = _read(sub / "plan.md")
        if not plan:
            continue
        script = ROOT / "scripts" / f"run_submission_{rec.date}_s{rec.num}.py"
        if not script.is_file():
            script = ROOT / "scripts" / f"run_submission_{rec.date}.py"
        if not script.is_file():
            continue
        prompt_parts = [
            f"You are {ADAPTER_NAMES['implement']}. {ADAPTER_GOALS['implement']}",
            f"Implement run script for submission-{rec.num} ({rec.date}).",
            f"## plan.md\n{plan[:8000]}",
        ]
        strategy = _read(sub / "strategy.md")
        if strategy:
            prompt_parts.append(f"## strategy.md\n{strategy[:4000]}")
        prompt_parts.append(summarize_arcgen_role(sub))
        prompt = "\n\n".join(prompt_parts)
        response = script.read_text(encoding="utf-8")
        row = {
            "adapter": ADAPTER_NAMES["implement"],
            "submission_dir": rec.rel(),
            "inputs": {"prompt": prompt},
            "output": response,
        }
        _write_example("implement", sub, row)
        texts.append(format_instruction(prompt, response))
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap LoRA examples from kaggle-submissions/")
    parser.add_argument("--adapter", choices=["diagnose", "strategize", "implement", "all"], default="all")
    args = parser.parse_args()

    records = discover_submissions()
    print(f"Found {len(records)} submission folders")
    for r in records:
        ka = f"{r.kaggle_actual:.2f}" if r.kaggle_actual is not None else "—"
        print(f"  {r.date} submission-{r.num}: pass_all={r.pass_all} kaggle={ka}")

    adapters = ["diagnose", "strategize", "implement"] if args.adapter == "all" else [args.adapter]
    for name in adapters:
        if name == "diagnose":
            texts = bootstrap_diagnose(records)
        elif name == "strategize":
            texts = bootstrap_strategize(records)
        else:
            texts = bootstrap_implement(records)
        if not texts:
            print(f"WARN: no {name} examples")
            continue
        export_mlx_jsonl(name, texts, ROOT / "training" / f"lora-{name}" / "mlx")

    print("Done. Run: python scripts/train_lora.py --adapter all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared LoRA dataset helpers: score timeline, log summaries, MLX JSONL export."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "kaggle-submissions"

ADAPTER_NAMES = {
    "diagnose": "NeuroGolf-Diagnose",
    "strategize": "NeuroGolf-Strategize",
    "implement": "NeuroGolf-Implement",
}

# Every adapter's north star: move the Kaggle public score up vs the prior scored submission.
SCORE_GOAL = (
    "PRIMARY GOAL: increase the Kaggle public score on neurogolf-2026. "
    "Diagnose what blocked score gains; strategize concrete levers (pass_all, verified tasks, "
    "ARC-GEN gating); implement minimal solver changes that beat the prior submission. "
    "Reject plans that are flat or regressive vs the latest scored baseline."
)

ADAPTER_GOALS = {
    "diagnose": (
        f"{SCORE_GOAL} Output: ranked failure modes, score delta vs prior, and which task families "
        "to fix first for maximum Kaggle gain."
    ),
    "strategize": (
        f"{SCORE_GOAL} Output: plan.md with expected pass_all delta and why Kaggle score should rise."
    ),
    "implement": (
        f"{SCORE_GOAL} Output: run script + arc_genome diff that implements the plan with measurable gates."
    ),
}

SUBMISSION_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})/submission-(\d+)$")


@dataclass
class SubmissionRecord:
    date: str
    num: int
    path: Path
    pass_all: int | None
    kaggle_actual: float | None
    kaggle_est: float | None
    message: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.date, self.num)

    def rel(self) -> str:
        return str(self.path.relative_to(ROOT))


def parse_submission_dir(path: Path) -> tuple[str, int] | None:
    if path.parent.parent != SUBMISSIONS:
        return None
    m = SUBMISSION_RE.match(f"{path.parent.name}/{path.name}")
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_results(sub_dir: Path) -> dict[str, Any]:
    p = sub_dir / "results.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def discover_submissions() -> list[SubmissionRecord]:
    rows: list[SubmissionRecord] = []
    if not SUBMISSIONS.is_dir():
        return rows
    for date_dir in sorted(SUBMISSIONS.iterdir()):
        if not date_dir.is_dir() or not re.match(r"\d{4}-\d{2}-\d{2}$", date_dir.name):
            continue
        for sub_dir in sorted(date_dir.glob("submission-*")):
            if not sub_dir.is_dir():
                continue
            parsed = parse_submission_dir(sub_dir)
            if not parsed:
                continue
            date, num = parsed
            r = load_results(sub_dir)
            rows.append(
                SubmissionRecord(
                    date=date,
                    num=num,
                    path=sub_dir,
                    pass_all=r.get("pass_all"),
                    kaggle_actual=r.get("kaggle_score_actual"),
                    kaggle_est=r.get("kaggle_score_est"),
                    message=r.get("message", ""),
                )
            )
    rows.sort(key=lambda x: (x.date, x.num))
    return rows


def score_delta_label(prev: SubmissionRecord | None, cur: SubmissionRecord) -> str:
    if prev is None or cur.kaggle_actual is None:
        return "baseline"
    if prev.kaggle_actual is None:
        return "first_scored"
    d = cur.kaggle_actual - prev.kaggle_actual
    if d > 1.0:
        return "score_up"
    if d < -1.0:
        return "score_down"
    return "score_flat"


def summarize_kaggle_logs(log_path: Path, max_tasks: int = 40) -> str:
    if not log_path.is_file():
        return ""
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    lines = ["# Kaggle logs summary"]
    summary = data.get("summary", {})
    if summary:
        lines.append(
            f"tasks_scored={summary.get('tasks_scored')} "
            f"kaggle_actual={summary.get('kaggle_actual')} "
            f"audit_ratio={summary.get('audit_ratio')}"
        )
    tasks = data.get("tasks", [])
    fails = [t for t in tasks if not t.get("pass_all")]
    passes = [t for t in tasks if t.get("pass_all")]
    lines.append(f"pass_all_count={len(passes)} fail_or_partial={len(fails)}")
    if fails:
        lines.append("## Tasks not pass_all (sample)")
        for t in fails[:max_tasks]:
            lines.append(
                f"- task {t.get('task')}: train_ok={t.get('train_ok')} "
                f"score={t.get('kaggle_score')}"
            )
    return "\n".join(lines)


def summarize_arcgen_role(sub_dir: Path) -> str:
    """ARC-GEN is used by solvers, not as raw LLM text — summarize how this submission used it."""
    r = load_results(sub_dir)
    n = r.get("arcgen_validate_samples")
    phase = r.get("phase")
    new_tasks = r.get("new_tasks") or r.get("prescan_new_candidates")
    lines = [
        "# ARC-GEN usage (solver validation, not LLM training text)",
        f"phase={phase}",
        f"arcgen_validate_samples={n}",
        f"new_tasks={new_tasks}",
        "ARC-GEN JSON grids train ONNX solvers via arc_genome; for LoRA we use "
        "Kaggle per-task logs and audit outcomes as the learning signal.",
    ]
    audit = sub_dir / "audit.json"
    if audit.is_file():
        try:
            a = json.loads(audit.read_text(encoding="utf-8"))
            b = a.get("summary", {}).get("buckets", {})
            lines.append(f"audit_buckets={json.dumps(b)}")
        except (json.JSONDecodeError, OSError):
            pass
    return "\n".join(lines)


def format_instruction(prompt: str, response: str) -> str:
    return (
        f"### Instruction:\n{prompt.strip()}\n\n"
        f"### Response:\n{response.strip()}"
    )


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_valid(texts: list[str], valid_frac: float = 0.2) -> tuple[list[str], list[str]]:
    if len(texts) <= 1:
        return texts, texts
    n_valid = max(1, int(len(texts) * valid_frac))
    if n_valid >= len(texts):
        n_valid = 1
    return texts[:-n_valid], texts[-n_valid:]


def export_mlx_jsonl(adapter: str, texts: list[str], out_dir: Path) -> None:
    train, valid = split_train_valid(texts)
    write_jsonl(out_dir / "train.jsonl", [{"text": t} for t in train])
    write_jsonl(out_dir / "valid.jsonl", [{"text": t} for t in valid])
    print(f"  mlx {adapter}: train={len(train)} valid={len(valid)} → {out_dir}")

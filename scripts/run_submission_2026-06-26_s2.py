#!/usr/bin/env python3
"""2026-06-26 submission-2: Phase 17 bbox-relative ARC-GEN gather."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import get_config, set_phase
from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
from arc_genome.genome.ops.arcgen_gather import s_bbox_gather_arcgen
from arc_genome.onnx.model import save_model
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
DATE = "2026-06-26"
SUB_NUM = 2
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"

LOGGED_DATE = "2026-06-17"
LOGGED_SUB_NUM = 4
LOGGED_SUB_DIR = f"kaggle-submissions/{LOGGED_DATE}/submission-{LOGGED_SUB_NUM}"
PRIOR_DATE = "2026-06-26"
PRIOR_SUB_NUM = 1
PRIOR_SUB_DIR = f"kaggle-submissions/{PRIOR_DATE}/submission-{PRIOR_SUB_NUM}"
SEED_DIR = f"{PRIOR_SUB_DIR}/submission"
SEED_ZIP = f"{PRIOR_SUB_DIR}/submission_v2.zip"
BASELINE_TASKS = 70
BASELINE_EST = 1065.5174872118832
BASELINE_SCORE = 915.03
SUBMIT_EST_MIN = BASELINE_EST + 1.0


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def _official_logged_tasks() -> set[int]:
    log_dir = f"{LOGGED_SUB_DIR}/kaggle_logs/official_tasks"
    solved: set[int] = set()
    if not os.path.isdir(log_dir):
        return solved
    for filename in os.listdir(log_dir):
        match = re.match(r"task(\d+)\.json$", filename)
        if match:
            solved.add(int(match.group(1)))
    return solved


def ensure_seed_dir() -> bool:
    if os.path.isdir(SEED_DIR) and any(f.endswith(".onnx") for f in os.listdir(SEED_DIR)):
        return True
    if not os.path.isfile(SEED_ZIP):
        return False
    os.makedirs(SEED_DIR, exist_ok=True)
    with zipfile.ZipFile(SEED_ZIP) as zf:
        zf.extractall(SEED_DIR)
    return True


def baseline_solved_tasks() -> set[int]:
    solved: set[int] = set()
    if ensure_seed_dir():
        for filename in os.listdir(SEED_DIR):
            if filename.endswith(".onnx"):
                solved.add(int(filename.replace("task", "").replace(".onnx", "")))
    solved.update(_official_logged_tasks())
    return solved


def prescan_new_tasks() -> list[int]:
    solved = baseline_solved_tasks()
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    new = []
    with tempfile.TemporaryDirectory() as tmp:
        for tn, meta in tasks.items():
            if tn in solved:
                continue
            td = meta["data"]
            model = s_bbox_gather_arcgen(td)
            if model is None:
                continue
            path = os.path.join(tmp, f"task{tn:03d}.onnx")
            save_model(model, path)
            if validate_full(path, td):
                new.append(tn)
    return sorted(new)


def seed_baseline(out_dir: str) -> int:
    if not ensure_seed_dir():
        return 0
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for filename in os.listdir(SEED_DIR):
        if filename.endswith(".onnx"):
            shutil.copy2(os.path.join(SEED_DIR, filename), os.path.join(out_dir, filename))
            copied += 1
    return copied


def write_results(results_doc: dict) -> None:
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)


def main():
    cfg = set_phase(17)
    if cfg.arcgen_validate_samples < 100:
        raise RuntimeError("Phase 17 must keep arcgen_validate_samples >= 100")
    if not cfg.arcgen_fit_bbox_gather:
        raise RuntimeError("Phase 17 must enable arcgen_fit_bbox_gather")
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    seeded = seed_baseline(OUT)
    print(f"Seeded {seeded} ONNX from {PRIOR_SUB_DIR}")

    new_candidates = prescan_new_tasks()
    print(f"Pre-scan new pass_all: {len(new_candidates)} {new_candidates}")

    log_path = f"{SUB_DIR}/run.log"
    start = time.time()

    with open(log_path, "w") as log_f:
        with contextlib.redirect_stdout(_Tee(sys.__stdout__, log_f)):
            print(f"Submission-{SUB_NUM}: Phase 17 bbox-relative ARC-GEN gather")
            results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
            make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")

    audit_path = f"{SUB_DIR}/audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_submission.py",
            "--submission_dir",
            OUT,
            "--out",
            audit_path,
            "--phase",
            "17",
        ],
        check=True,
    )

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    buckets = summary["buckets"]
    pass_all = buckets["pass_all"]
    train_only = buckets.get("train_only", 0)
    est = summary["pass_all_kaggle_total"]
    solver_counts = Counter(r["solver"] for r in results.values())

    results_doc = {
        "submission": SUB_NUM,
        "date": DATE,
        "phase": 17,
        "milestone": "M10b",
        "solved": len(results),
        "pass_all": pass_all,
        "train_only": train_only,
        "kaggle_score_est": est,
        "kaggle_score_baseline": BASELINE_SCORE,
        "kaggle_score_est_baseline": BASELINE_EST,
        "baseline_submission_tasks": BASELINE_TASKS,
        "baseline_logged_dir": LOGGED_SUB_DIR,
        "prescan_new_candidates": new_candidates,
        "seeded_from": SEED_ZIP if seeded else None,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(solver_counts.most_common()),
        "arcgen_validate_samples": get_config().arcgen_validate_samples,
        "elapsed_s": round(time.time() - start, 1),
    }
    write_results(results_doc)

    should_submit = train_only == 0 and (pass_all > BASELINE_TASKS or est >= SUBMIT_EST_MIN)
    print(
        f"\n=== Submit: pass_all={pass_all}, train_only={train_only}, "
        f"est={est:.1f}, submit={should_submit} ==="
    )

    if not should_submit:
        results_doc["message"] = f"No submit: {pass_all} pass_all, {train_only} train_only, est {est:.0f}"
        write_results(results_doc)
        notes = (
            f"# submission-2 notes\n\n"
            f"Blocked submit: pass_all={pass_all} (need >{BASELINE_TASKS}), "
            f"train_only={train_only}, est={est:.1f} (need >={SUBMIT_EST_MIN:.1f}).\n"
            f"prescan_new={new_candidates}\n"
        )
        with open(f"{SUB_DIR}/notes.md", "w") as f:
            f.write(notes)
        sys.exit(0)

    if os.environ.get("NEUROGOLF_SKIP_KAGGLE_SUBMIT"):
        results_doc["message"] = f"Solve push only: {pass_all} pass_all, est {est:.0f}"
        results_doc["ready_for_manual_kaggle_submit"] = True
        write_results(results_doc)
        print("NEUROGOLF_SKIP_KAGGLE_SUBMIT set; zip ready, no Kaggle CLI submit.")
        sys.exit(0)

    msg = f"ARC-Genome M10b: {pass_all} verified, est {est:.0f}, bbox gather arcgen"
    shutil.copy2(ZIP_STORE, SUBMIT_PATH)
    proc = subprocess.run(
        [
            KAGGLE,
            "competitions",
            "submit",
            "-c",
            "neurogolf-2026",
            "-f",
            SUBMIT_PATH,
            "-m",
            msg,
        ]
    )
    if proc.returncode != 0:
        sys.exit(1)
    results_doc["submitted"] = True
    results_doc["message"] = msg
    from datetime import datetime, timezone

    results_doc["submitted_at"] = datetime.now(timezone.utc).isoformat()
    write_results(results_doc)
    print("Submitted successfully.")


if __name__ == "__main__":
    main()

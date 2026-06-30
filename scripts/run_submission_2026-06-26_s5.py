#!/usr/bin/env python3
"""2026-06-26 submission-5: compact slim gravity for tasks 32/78 (Kaggle-size safe)."""

from __future__ import annotations

import contextlib
import json
import os
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
from arc_genome.genome.ops.arcgen_gravity import (
    ARCgen_GRAVITY_SOLVERS,
    s_gravity_down_dynamic,
    s_gravity_up_dynamic,
)
from arc_genome.onnx.model import ONNX_MAX_BYTES, onnx_file_size_ok, save_model
from arc_genome.solve import make_submission_zip

DATE = "2026-06-26"
SUB_NUM = 5
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"

PRIOR_SUB_DIR = "kaggle-submissions/2026-06-26/submission-2"
SEED_DIR = f"{PRIOR_SUB_DIR}/submission"
SEED_ZIP = f"{PRIOR_SUB_DIR}/submission_v2.zip"
BASELINE_TASKS = 72
BASELINE_SCORE = 940.75
BASELINE_EST = 1091.2298643193174
SUBMIT_EST_MIN = BASELINE_EST + 1.0

TARGET_TASKS = {
    32: ("gravity_down_dynamic", s_gravity_down_dynamic),
    78: ("gravity_up_dynamic", s_gravity_up_dynamic),
}


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def ensure_seed_dir() -> bool:
    if os.path.isdir(SEED_DIR) and any(f.endswith(".onnx") for f in os.listdir(SEED_DIR)):
        return True
    if not os.path.isfile(SEED_ZIP):
        return False
    os.makedirs(SEED_DIR, exist_ok=True)
    with zipfile.ZipFile(SEED_ZIP) as zf:
        zf.extractall(SEED_DIR)
    return True


def seed_baseline(out_dir: str) -> int:
    if not ensure_seed_dir():
        return 0
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for filename in os.listdir(SEED_DIR):
        if not filename.endswith(".onnx"):
            continue
        src = os.path.join(SEED_DIR, filename)
        if not onnx_file_size_ok(src):
            continue
        shutil.copy2(src, os.path.join(out_dir, filename))
        copied += 1
    return copied


def prescan_new_tasks() -> list[int]:
    if not ensure_seed_dir():
        return []
    solved = {
        int(f.replace("task", "").replace(".onnx", ""))
        for f in os.listdir(SEED_DIR)
        if f.endswith(".onnx")
    }
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    new: set[int] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for tn, meta in tasks.items():
            if tn in solved:
                continue
            td = meta["data"]
            for _name, sfn in ARCgen_GRAVITY_SOLVERS:
                model = sfn(td)
                if model is None:
                    continue
                path = os.path.join(tmp, f"task{tn:03d}.onnx")
                save_model(model, path)
                if validate_full(path, td) and onnx_file_size_ok(path):
                    new.add(tn)
                    break
    return sorted(new)


def compile_targets(out_dir: str) -> dict[int, str]:
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    compiled: dict[int, str] = {}
    for tn, (sname, sfn) in TARGET_TASKS.items():
        td = tasks[tn]["data"]
        model = sfn(td)
        if model is None:
            raise RuntimeError(f"Failed to compile task {tn}")
        path = os.path.join(out_dir, f"task{tn:03d}.onnx")
        save_model(model, path)
        if not validate_full(path, td):
            raise RuntimeError(f"validate_full failed for task {tn}")
        if not onnx_file_size_ok(path):
            raise RuntimeError(f"Task {tn} ONNX exceeds {ONNX_MAX_BYTES} bytes")
        compiled[tn] = sname
        print(f"Compiled task {tn}: {sname} ({os.path.getsize(path)/1024:.1f} KB)")
    return compiled


def write_results(results_doc: dict) -> None:
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)


def main():
    cfg = set_phase(22)
    if cfg.arcgen_validate_samples < 100:
        raise RuntimeError("Must keep arcgen_validate_samples >= 100")
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    start = time.time()
    log_path = f"{SUB_DIR}/run.log"
    with open(log_path, "w") as log_f:
        with contextlib.redirect_stdout(_Tee(sys.__stdout__, log_f)):
            seeded = seed_baseline(OUT)
            print(f"Seeded {seeded} ONNX from {PRIOR_SUB_DIR} (size-safe only)")

            new_candidates = prescan_new_tasks()
            print(f"Pre-scan compact gravity: {new_candidates}")

            compiled = compile_targets(OUT)
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
            "22",
        ],
        check=True,
    )

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    buckets = summary["buckets"]
    pass_all = buckets["pass_all"]
    kaggle_eligible = buckets.get("kaggle_eligible", pass_all)
    oversized = buckets.get("oversized", 0)
    train_only = buckets.get("train_only", 0)
    est = summary["pass_all_kaggle_total"]
    est_eligible = summary.get("kaggle_eligible_total", est)

    results_doc = {
        "submission": SUB_NUM,
        "date": DATE,
        "phase": 23,
        "milestone": "M13",
        "solved": pass_all,
        "pass_all": pass_all,
        "kaggle_eligible": kaggle_eligible,
        "oversized_pass_all": oversized,
        "train_only": train_only,
        "kaggle_score_est": est,
        "kaggle_score_est_eligible": est_eligible,
        "kaggle_score_baseline": BASELINE_SCORE,
        "kaggle_score_est_baseline": BASELINE_EST,
        "baseline_submission_tasks": BASELINE_TASKS,
        "baseline_logged_dir": PRIOR_SUB_DIR,
        "prescan_new_candidates": new_candidates,
        "new_tasks": sorted(compiled.keys()),
        "seeded_from": SEED_ZIP,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": compiled,
        "arcgen_validate_samples": get_config().arcgen_validate_samples,
        "elapsed_s": round(time.time() - start, 1),
    }
    write_results(results_doc)

    should_submit = (
        train_only == 0
        and oversized == 0
        and kaggle_eligible > BASELINE_TASKS
        and est_eligible >= SUBMIT_EST_MIN
    )
    print(
        f"\n=== Submit: kaggle_eligible={kaggle_eligible}, pass_all={pass_all}, "
        f"oversized={oversized}, est_eligible={est_eligible:.1f}, submit={should_submit} ==="
    )

    if not should_submit:
        results_doc["message"] = (
            f"No submit: kaggle_eligible={kaggle_eligible} (need >{BASELINE_TASKS}), "
            f"oversized={oversized}, est_eligible={est_eligible:.0f}"
        )
        write_results(results_doc)
        with open(f"{SUB_DIR}/notes.md", "w") as f:
            f.write(f"# submission-5 notes\n\nBlocked: {results_doc['message']}\n")
        sys.exit(0)

    msg = f"ARC-Genome M13: {kaggle_eligible} verified, est {est_eligible:.0f}, compact gravity"
    results_doc["message"] = msg
    results_doc["ready_for_manual_kaggle_submit"] = True
    write_results(results_doc)

    if os.environ.get("NEUROGOLF_SKIP_KAGGLE_SUBMIT"):
        subprocess.run(
            [
                sys.executable,
                "scripts/write_kaggle_submit_ready.py",
                "--submission-dir",
                SUB_DIR,
                "--message",
                msg,
            ],
            check=False,
        )
        print("NEUROGOLF_SKIP_KAGGLE_SUBMIT set; kaggle_submit_ready.json for GHA.")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""2026-06-17 submission-2: Phase 13 ARC-GEN routing + bounded upscale2."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import set_phase
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
DATE = "2026-06-17"
SUB_NUM = 2
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"
BASELINE_TASKS = 64
BASELINE_SCORE = 835.12


def prescan_new_tasks() -> list[int]:
    """Numpy pre-scan: tasks with new bounded_upscale2 pass_all potential."""
    import tempfile
    from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
    from arc_genome.genome.ops.bounded import s_bounded_upscale2
    from arc_genome.onnx.model import save_model

    s1 = f"kaggle-submissions/{DATE}/submission-1/submission"
    solved = set()
    if os.path.isdir(s1):
        for f in os.listdir(s1):
            if f.endswith(".onnx"):
                solved.add(int(f.replace("task", "").replace(".onnx", "")))

    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    new_tasks = []
    with tempfile.TemporaryDirectory() as tmp:
        for tn, meta in tasks.items():
            if tn in solved:
                continue
            td = meta["data"]
            model = s_bounded_upscale2(td)
            if model is None:
                continue
            p = os.path.join(tmp, f"task{tn:03d}.onnx")
            save_model(model, p)
            if validate_full(p, td):
                new_tasks.append(tn)
    return new_tasks


def main():
    set_phase(13)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    print("=== Pre-scan (ARC-GEN routing candidates) ===")
    new_candidates = prescan_new_tasks()
    print(f"New pass_all candidates: {len(new_candidates)} {new_candidates}")

    log_path = f"{SUB_DIR}/run.log"
    t0 = time.time()
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, s):
            for st in self._streams:
                st.write(s)
        def flush(self):
            for st in self._streams:
                st.flush()

    log_f = open(log_path, "w")
    import contextlib
    with contextlib.redirect_stdout(_Tee(sys.__stdout__, log_f)):
        print(f"Submission-{SUB_NUM} {DATE}: Phase 13 ARC-GEN routing...")
        results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
        make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")
    log_f.close()

    audit_path = f"{SUB_DIR}/audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT,
        "--out", audit_path,
        "--phase", "13",
    ], check=True)

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    from collections import Counter
    sc = Counter(r["solver"] for r in results.values())
    n = summary["buckets"]["pass_all"]
    est = summary["pass_all_kaggle_total"]
    elapsed = time.time() - t0

    results_doc = {
        "submission": SUB_NUM,
        "date": DATE,
        "phase": 13,
        "milestone": "M8",
        "solved": len(results),
        "pass_all": n,
        "kaggle_score_est": est,
        "prescan_new_candidates": new_candidates,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(sc.most_common()),
        "baseline_submission1_tasks": BASELINE_TASKS,
        "baseline_submission1_kaggle": BASELINE_SCORE,
        "elapsed_s": round(elapsed, 1),
    }
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    should_submit = n > BASELINE_TASKS or (n == BASELINE_TASKS and est > BASELINE_SCORE * 0.99)
    print(f"\n=== Kaggle submit decision: pass_all={n}, est={est:.1f}, submit={should_submit} ===")

    if not should_submit:
        print("No submit: no improvement over submission-1 baseline.")
        results_doc["message"] = f"No submit: {n} tasks, est {est:.0f}"
        with open(f"{SUB_DIR}/results.json", "w") as f:
            json.dump(results_doc, f, indent=2)
        sys.exit(0)

    msg = f"ARC-Genome M8: {n} verified, est {est:.0f}, arcgen routing"
    shutil.copy2(ZIP_STORE, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        print("Submit failed")
        sys.exit(1)

    results_doc["submitted"] = True
    results_doc["message"] = msg
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    status_proc = subprocess.run(
        [KAGGLE, "competitions", "submissions", "list", "-c", "neurogolf-2026", "--csv"],
        capture_output=True, text=True,
    )
    if status_proc.returncode == 0:
        with open(f"{SUB_DIR}/kaggle_status.txt", "w") as f:
            f.write(status_proc.stdout)

    print("Submitted successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()

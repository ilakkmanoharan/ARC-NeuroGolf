#!/usr/bin/env python3
"""2026-06-17 submission-3: Phase 14 autodiscover + bounded library expansion."""

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
SUB_NUM = 3
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"
BASELINE_TASKS = 65
BASELINE_SCORE = 848.07
BASELINE_EST = 998.55
SEED_DIR = f"kaggle-submissions/{DATE}/submission-2/submission"


def prescan_new_tasks() -> list[int]:
    from arc_genome.data.arcgen import load_tasks_with_arcgen
    from arc_genome.genome.ops.autodiscover import prescan_new_tasks as scan

    solved = set()
    if os.path.isdir(SEED_DIR):
        for f in os.listdir(SEED_DIR):
            if f.endswith(".onnx"):
                solved.add(int(f.replace("task", "").replace(".onnx", "")))
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    return scan(solved, tasks)


def seed_baseline(out_dir: str) -> int:
    """Copy submission-2 ONNX as cost_audit baseline."""
    if not os.path.isdir(SEED_DIR):
        return 0
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in os.listdir(SEED_DIR):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(SEED_DIR, f), os.path.join(out_dir, f))
            n += 1
    return n


def main():
    set_phase(14)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    seeded = seed_baseline(OUT)
    print(f"Seeded {seeded} ONNX files from submission-2")

    print("=== Pre-scan (autodiscover bounded rules) ===")
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
        print(f"Submission-{SUB_NUM} {DATE}: Phase 14 autodiscover...")
        results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
        make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")
    log_f.close()

    audit_path = f"{SUB_DIR}/audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT,
        "--out", audit_path,
        "--phase", "14",
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
        "phase": 14,
        "milestone": "M8b",
        "solved": len(results),
        "pass_all": n,
        "kaggle_score_est": est,
        "kaggle_score_baseline": BASELINE_SCORE,
        "prescan_new_candidates": new_candidates,
        "seeded_from": SEED_DIR,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(sc.most_common()),
        "elapsed_s": round(elapsed, 1),
    }
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    should_submit = n >= BASELINE_TASKS and (n > BASELINE_TASKS or est >= BASELINE_EST - 0.1)
    print(f"\n=== Submit decision: pass_all={n}, est={est:.1f}, submit={should_submit} ===")

    if not should_submit:
        print("No submit: no improvement over submission-2.")
        results_doc["message"] = f"No submit: {n} tasks, est {est:.0f}"
        with open(f"{SUB_DIR}/results.json", "w") as f:
            json.dump(results_doc, f, indent=2)
        sys.exit(0)

    msg = f"ARC-Genome M8b: {n} verified, est {est:.0f}, autodiscover"
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
    print("Submitted successfully.")


if __name__ == "__main__":
    main()

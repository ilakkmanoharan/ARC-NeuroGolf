#!/usr/bin/env python3
"""2026-06-17 submission-4: Phase 15 ARC-GEN-fitted gather (+coverage to 70)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import set_phase
from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
from arc_genome.genome.ops.arcgen_gather import s_position_gather_arcgen
from arc_genome.onnx.model import save_model
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
DATE = "2026-06-17"
SUB_NUM = 4
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"
SEED_DIR = f"kaggle-submissions/{DATE}/submission-3/submission"
BASELINE_TASKS = 65
BASELINE_SCORE = 848.07
BASELINE_EST = 998.55


def prescan_new_tasks() -> list[int]:
    solved = set()
    if os.path.isdir(SEED_DIR):
        for f in os.listdir(SEED_DIR):
            if f.endswith(".onnx"):
                solved.add(int(f.replace("task", "").replace(".onnx", "")))
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    new = []
    with tempfile.TemporaryDirectory() as tmp:
        for tn, meta in tasks.items():
            if tn in solved:
                continue
            td = meta["data"]
            model = s_position_gather_arcgen(td)
            if model is None:
                continue
            p = os.path.join(tmp, f"task{tn:03d}.onnx")
            save_model(model, p)
            if validate_full(p, td):
                new.append(tn)
    return sorted(new)


def seed_baseline(out_dir: str) -> int:
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
    set_phase(15)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    seeded = seed_baseline(OUT)
    print(f"Seeded {seeded} ONNX from submission-3")

    new_candidates = prescan_new_tasks()
    print(f"Pre-scan new pass_all: {len(new_candidates)} {new_candidates}")

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
        print(f"Submission-{SUB_NUM}: Phase 15 arcgen-fitted gather")
        results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
        make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")
    log_f.close()

    audit_path = f"{SUB_DIR}/audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT, "--out", audit_path, "--phase", "15",
    ], check=True)

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    from collections import Counter
    sc = Counter(r["solver"] for r in results.values())
    n = summary["buckets"]["pass_all"]
    est = summary["pass_all_kaggle_total"]

    results_doc = {
        "submission": SUB_NUM,
        "date": DATE,
        "phase": 15,
        "milestone": "M9",
        "solved": len(results),
        "pass_all": n,
        "kaggle_score_est": est,
        "kaggle_score_baseline": BASELINE_SCORE,
        "prescan_new_candidates": new_candidates,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(sc.most_common()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    should_submit = n > BASELINE_TASKS or est > BASELINE_EST + 1.0
    print(f"\n=== Submit: pass_all={n}, est={est:.1f}, submit={should_submit} ===")

    if not should_submit:
        results_doc["message"] = f"No submit: {n} tasks, est {est:.0f}"
        with open(f"{SUB_DIR}/results.json", "w") as f:
            json.dump(results_doc, f, indent=2)
        sys.exit(0)

    msg = f"ARC-Genome M9: {n} verified, est {est:.0f}, arcgen gather fit"
    shutil.copy2(ZIP_STORE, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        sys.exit(1)
    results_doc["submitted"] = True
    results_doc["message"] = msg
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)
    print("Submitted successfully.")


if __name__ == "__main__":
    main()

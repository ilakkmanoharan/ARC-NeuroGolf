#!/usr/bin/env python3
"""2026-06-17 submission-5: Phase 16 ARC-GEN-fitted color maps."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_genome.config import get_config, set_phase
from arc_genome.data.arcgen import load_tasks_with_arcgen, validate_full
from arc_genome.genome.ops.analytical import s_color_map_arcgen
from arc_genome.onnx.model import save_model
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
DATE = "2026-06-17"
SUB_NUM = 5
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"
SEED_DIR = f"kaggle-submissions/{DATE}/submission-4/submission"
SEED_ZIP = f"kaggle-submissions/{DATE}/submission-4/submission_v2.zip"
BASELINE_TASKS = 70
BASELINE_EST = 1065.5174872118832
BASELINE_SCORE = 915.03


def _official_logged_tasks() -> set[int]:
    log_dir = f"kaggle-submissions/{DATE}/submission-4/kaggle_logs/official_tasks"
    solved: set[int] = set()
    if not os.path.isdir(log_dir):
        return solved
    for f in os.listdir(log_dir):
        m = re.match(r"task(\d+)\.json$", f)
        if m:
            solved.add(int(m.group(1)))
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
        for f in os.listdir(SEED_DIR):
            if f.endswith(".onnx"):
                solved.add(int(f.replace("task", "").replace(".onnx", "")))
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
            model = s_color_map_arcgen(td)
            if model is None:
                continue
            p = os.path.join(tmp, f"task{tn:03d}.onnx")
            save_model(model, p)
            if validate_full(p, td):
                new.append(tn)
    return sorted(new)


def seed_baseline(out_dir: str) -> int:
    if not ensure_seed_dir():
        return 0
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in os.listdir(SEED_DIR):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(SEED_DIR, f), os.path.join(out_dir, f))
            n += 1
    return n


def main():
    cfg = set_phase(16)
    if cfg.arcgen_validate_samples < 100:
        raise RuntimeError("Phase 16 must keep arcgen_validate_samples >= 100")
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    seeded = seed_baseline(OUT)
    print(f"Seeded {seeded} ONNX from submission-4")

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
        print(f"Submission-{SUB_NUM}: Phase 16 arcgen-fitted color maps")
        results = solve_all("data/all_tasks.json", OUT, conv_budget=25)
        make_submission_zip(OUT, ZIP_STORE, "data/all_tasks.json")
    log_f.close()

    audit_path = f"{SUB_DIR}/audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_submission.py",
        "--submission_dir", OUT, "--out", audit_path, "--phase", "16",
    ], check=True)

    with open(audit_path) as f:
        audit = json.load(f)
    summary = audit["summary"]
    from collections import Counter
    sc = Counter(r["solver"] for r in results.values())
    buckets = summary["buckets"]
    n = buckets["pass_all"]
    train_only = buckets.get("train_only", 0)
    est = summary["pass_all_kaggle_total"]

    results_doc = {
        "submission": SUB_NUM,
        "date": DATE,
        "phase": 16,
        "milestone": "M10a",
        "solved": len(results),
        "pass_all": n,
        "train_only": train_only,
        "kaggle_score_est": est,
        "kaggle_score_baseline": BASELINE_SCORE,
        "kaggle_score_est_baseline": BASELINE_EST,
        "prescan_new_candidates": new_candidates,
        "seeded_from": SEED_DIR if seeded else None,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(sc.most_common()),
        "arcgen_validate_samples": get_config().arcgen_validate_samples,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)

    should_submit = train_only == 0 and (n > BASELINE_TASKS or est >= BASELINE_EST + 1.0)
    print(
        f"\n=== Submit: pass_all={n}, train_only={train_only}, "
        f"est={est:.1f}, submit={should_submit} ==="
    )

    if not should_submit:
        results_doc["message"] = f"No submit: {n} pass_all, {train_only} train_only, est {est:.0f}"
        with open(f"{SUB_DIR}/results.json", "w") as f:
            json.dump(results_doc, f, indent=2)
        sys.exit(0)

    if os.environ.get("NEUROGOLF_SKIP_KAGGLE_SUBMIT"):
        results_doc["message"] = f"Solve push only: {n} pass_all, est {est:.0f}"
        results_doc["ready_for_manual_kaggle_submit"] = True
        with open(f"{SUB_DIR}/results.json", "w") as f:
            json.dump(results_doc, f, indent=2)
        print("NEUROGOLF_SKIP_KAGGLE_SUBMIT set — zip ready, no Kaggle CLI submit.")
        sys.exit(0)

    msg = f"ARC-Genome M10a: {n} verified, est {est:.0f}, arcgen color fit"
    shutil.copy2(ZIP_STORE, SUBMIT_PATH)
    proc = subprocess.run([
        KAGGLE, "competitions", "submit", "-c", "neurogolf-2026",
        "-f", SUBMIT_PATH, "-m", msg,
    ])
    if proc.returncode != 0:
        sys.exit(1)
    results_doc["submitted"] = True
    results_doc["message"] = msg
    from datetime import datetime, timezone
    results_doc["submitted_at"] = datetime.now(timezone.utc).isoformat()
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)
    print("Submitted successfully.")


if __name__ == "__main__":
    main()

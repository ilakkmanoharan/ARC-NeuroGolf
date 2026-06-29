#!/usr/bin/env python3
"""2026-06-26 submission-4: Phase 22 depth-3 compose + full unsolved conv pass."""

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
from arc_genome.genome.ops.arcgen_compose import s_compose_arcgen
from arc_genome.genome.ops.arcgen_gravity import ARCgen_GRAVITY_SOLVERS
from arc_genome.genome.ops.arcgen_object import ARCgen_OBJECT_SOLVERS
from arc_genome.genome.ops.arcgen_place import ARCgen_PLACE_SOLVERS
from arc_genome.onnx.model import ONNX_MAX_BYTES, onnx_file_size_ok, save_model
from arc_genome.solve import make_submission_zip, solve_all

KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/Library/Python/3.9/bin/kaggle"))
DATE = "2026-06-26"
SUB_NUM = 4
SUB_DIR = f"kaggle-submissions/{DATE}/submission-{SUB_NUM}"
OUT = f"{SUB_DIR}/submission"
ZIP_STORE = f"{SUB_DIR}/submission_v2.zip"
SUBMIT_PATH = "submission.zip"

PRIOR_SUB_DIR = "kaggle-submissions/2026-06-26/submission-3"
SEED_DIR = f"{PRIOR_SUB_DIR}/submission"
SEED_ZIP = f"{PRIOR_SUB_DIR}/submission_v2.zip"
BASELINE_TASKS = 72
BASELINE_SCORE = 940.75
BASELINE_EST = 1091.2298643193174
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
    return solved


def prescan_new_tasks() -> list[int]:
    solved = baseline_solved_tasks()
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    new: set[int] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for tn, meta in tasks.items():
            if tn in solved:
                continue
            td = meta["data"]
            for _name, sfn in [
                *ARCgen_GRAVITY_SOLVERS,
                ("compose_arcgen", s_compose_arcgen),
                *ARCgen_PLACE_SOLVERS,
                *ARCgen_OBJECT_SOLVERS,
            ]:
                model = sfn(td)
                if model is None:
                    continue
                path = os.path.join(tmp, f"task{tn:03d}.onnx")
                save_model(model, path)
                if validate_full(path, td):
                    new.add(tn)
                    break
    return sorted(new)


def seed_baseline(out_dir: str) -> tuple[int, list[int]]:
    if not ensure_seed_dir():
        return 0, []
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    skipped_oversized: list[int] = []
    for filename in os.listdir(SEED_DIR):
        if not filename.endswith(".onnx"):
            continue
        src = os.path.join(SEED_DIR, filename)
        if not onnx_file_size_ok(src):
            skipped_oversized.append(int(filename.replace("task", "").replace(".onnx", "")))
            continue
        shutil.copy2(src, os.path.join(out_dir, filename))
        copied += 1
    if skipped_oversized:
        print(f"Skipped oversized seeds (>{ONNX_MAX_BYTES} B): {sorted(skipped_oversized)}")
    return copied, skipped_oversized


def write_results(results_doc: dict) -> None:
    with open(f"{SUB_DIR}/results.json", "w") as f:
        json.dump(results_doc, f, indent=2)


def main():
    cfg = set_phase(22)
    if cfg.arcgen_validate_samples < 100:
        raise RuntimeError("Phase 22 must keep arcgen_validate_samples >= 100")
    if cfg.arcgen_compose_depth < 3:
        raise RuntimeError("Phase 22 must enable arcgen_compose_depth >= 3")
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    seeded, skipped_oversized = seed_baseline(OUT)
    print(f"Seeded {seeded} ONNX from {PRIOR_SUB_DIR}")
    if skipped_oversized:
        print(f"Oversized seeds excluded from zip (Kaggle 1.44MB cap): {skipped_oversized}")

    baseline = baseline_solved_tasks()
    tasks = load_tasks_with_arcgen("data/all_tasks.json")
    unsolved = sorted(set(tasks.keys()) - baseline)
    print(f"Unsolved pool: {len(unsolved)} tasks")

    new_candidates = prescan_new_tasks()
    print(f"Pre-scan new pass_all: {len(new_candidates)} {new_candidates}")

    log_path = f"{SUB_DIR}/run.log"
    start = time.time()

    with open(log_path, "w") as log_f:
        with contextlib.redirect_stdout(_Tee(sys.__stdout__, log_f)):
            print(f"Submission-{SUB_NUM}: Phase 22 depth-3 compose + unsolved conv pass")
            results = solve_all("data/all_tasks.json", OUT, conv_budget=25, task_nums=unsolved)
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
    solver_counts = Counter(r["solver"] for r in results.values())
    new_tasks = sorted(set(results.keys()) - baseline)

    results_doc = {
        "submission": SUB_NUM,
        "date": DATE,
        "phase": 22,
        "milestone": "M12",
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
        "new_tasks": new_tasks,
        "seeded_from": SEED_ZIP if seeded else None,
        "skipped_oversized_seeds": skipped_oversized,
        "zip": ZIP_STORE,
        "submitted": False,
        "solver_counts": dict(solver_counts.most_common()),
        "arcgen_validate_samples": get_config().arcgen_validate_samples,
        "elapsed_s": round(time.time() - start, 1),
    }
    write_results(results_doc)

    should_submit = (
        train_only == 0
        and oversized == 0
        and (kaggle_eligible > BASELINE_TASKS or est_eligible >= SUBMIT_EST_MIN)
    )
    print(
        f"\n=== Submit: kaggle_eligible={kaggle_eligible}, pass_all={pass_all}, "
        f"oversized={oversized}, train_only={train_only}, "
        f"est_eligible={est_eligible:.1f}, submit={should_submit} ==="
    )

    if not should_submit:
        results_doc["message"] = (
            f"No submit: kaggle_eligible={kaggle_eligible} (need >{BASELINE_TASKS}), "
            f"oversized={oversized}, train_only={train_only}, est_eligible={est_eligible:.0f}"
        )
        write_results(results_doc)
        notes = (
            f"# submission-4 notes\n\n"
            f"Blocked submit: kaggle_eligible={kaggle_eligible} (need >{BASELINE_TASKS}), "
            f"oversized={oversized}, train_only={train_only}, est_eligible={est_eligible:.1f} "
            f"(need >={SUBMIT_EST_MIN:.1f}).\n"
            f"prescan_new={new_candidates}\n"
            f"new_from_solve={new_tasks}\n"
            f"skipped_oversized_seeds={skipped_oversized}\n"
        )
        with open(f"{SUB_DIR}/notes.md", "w") as f:
            f.write(notes)
        sys.exit(0)

    if os.environ.get("NEUROGOLF_SKIP_KAGGLE_SUBMIT"):
        results_doc["message"] = (
            f"ARC-Genome M12: {pass_all} verified, est {est:.0f}, unsolved conv pass"
        )
        results_doc["ready_for_manual_kaggle_submit"] = True
        write_results(results_doc)
        subprocess.run(
            [
                sys.executable,
                "scripts/write_kaggle_submit_ready.py",
                "--submission-dir",
                SUB_DIR,
                "--message",
                results_doc["message"],
            ],
            check=False,
        )
        print("NEUROGOLF_SKIP_KAGGLE_SUBMIT set; zip + kaggle_submit_ready.json for GHA.")
        sys.exit(0)

    msg = f"ARC-Genome M12: {kaggle_eligible} verified, est {est_eligible:.0f}, unsolved conv pass"
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

#!/usr/bin/env python3
"""Fine-tune LoRA adapters with MLX-LM (Apple Silicon efficient) or emit training commands."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Small instruct model — fast QLoRA on Mac; swap for larger when you have 20+ examples
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

ADAPTERS = ("diagnose", "strategize", "implement")


def _mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def train_one(
    adapter: str,
    model: str,
    iters: int,
    batch_size: int,
    lora_layers: int,
    learning_rate: float,
    dry_run: bool,
) -> int:
    data_dir = ROOT / "training" / f"lora-{adapter}" / "mlx"
    train_file = data_dir / "train.jsonl"
    if not train_file.is_file():
        print(f"Missing {train_file} — run: python scripts/bootstrap_lora_training_data.py", file=sys.stderr)
        return 1

    n_lines = sum(1 for _ in train_file.open())
    if n_lines < 2:
        print(f"WARN: only {n_lines} training rows for {adapter} — expect overfitting; collect more submissions")

    out_dir = ROOT / "training" / f"lora-{adapter}" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(out_dir),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(lora_layers),
        "--learning-rate",
        str(learning_rate),
        "--steps-per-report",
        "10",
        "--steps-per-eval",
        "50",
        "--val-batches",
        "4",
    ]

    print(f"\n=== {adapter} ({n_lines} train rows) ===")
    print(" ".join(cmd))
    if dry_run:
        return 0

    if not _mlx_available():
        print("Install: pip install -r requirements-training.txt", file=sys.stderr)
        return 1

    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NeuroGolf LoRA adapters (MLX-LM)")
    parser.add_argument("--adapter", choices=[*ADAPTERS, "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--iters", type=int, default=200, help="More iters when dataset grows (try 500+ at 20+ rows)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--bootstrap", action="store_true", help="Run bootstrap_lora_training_data.py first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.bootstrap:
        boot = ROOT / "scripts" / "bootstrap_lora_training_data.py"
        rc = subprocess.call([sys.executable, str(boot), "--adapter", args.adapter])
        if rc != 0:
            return rc

    adapters = list(ADAPTERS) if args.adapter == "all" else [args.adapter]
    rc = 0
    for name in adapters:
        rc |= train_one(
            name,
            args.model,
            args.iters,
            args.batch_size,
            args.lora_layers,
            args.learning_rate,
            args.dry_run,
        )
    if rc == 0 and not args.dry_run:
        print("\nAdapters saved under training/lora-*/checkpoints/")
        print("Inference: mlx_lm.generate --model", args.model, "--adapter-path training/lora-diagnose/checkpoints")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

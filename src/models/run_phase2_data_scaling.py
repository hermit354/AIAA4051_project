#!/usr/bin/env python3
"""Run Phase 2.5 Flan-T5-base data-scaling experiments.

Pipeline stage: orchestration for training-size ablation experiments.
Inputs: scaling datasets, model alias, percent levels, and output-root settings.
Outputs: configs, trained checkpoints, predictions, and metrics by data percentage.
Typical caller: scripts/run_phase2_data_scaling.sh.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json
from src.models.model_registry import model_path
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS


def command_prefix() -> list[str]:
    """Return a configurable Conda-backed Python command for child jobs."""
    conda_base = Path(os.environ.get("CONDA_BASE", str(Path.home() / "miniforge3")))
    conda_bin = Path(os.environ.get("CONDA_BIN", str(conda_base / "bin" / "conda")))
    env_name = os.environ.get("ENV_NAME", "NLP")
    return [str(conda_bin), "run", "-n", env_name, "python"]


def run_command(command: list[str]) -> None:
    """Print and run one subprocess command with failure propagation."""
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train_config(model_name: str, dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create a training config whose eval cadence scales with data size."""
    train_count = len(load_records(dataset_dir / "jsonl" / "train.jsonl"))
    batch_size = 16
    total_steps = math.ceil(train_count / batch_size) * 20
    eval_steps = max(25, min(500, max(1, total_steps // 10)))
    return {
        "model_name_or_path": str(model_path(model_name)),
        "output_dir": str(output_dir),
        "train_path": str(dataset_dir / "jsonl" / "train.jsonl"),
        "eval_path": str(dataset_dir / "jsonl" / "val.jsonl"),
        "max_source_length": 160,
        "max_target_length": 96,
        "num_train_epochs": 20,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.03,
        "eval_steps": eval_steps,
        "save_steps": eval_steps,
        "save_total_limit": 3,
        "early_stopping_patience": 8,
        "eval_max_new_tokens": 96,
        "num_beams": 1,
        "seed": 1,
        "fp16": False,
        "bf16": True,
    }


def has_completed_training(output_dir: Path) -> bool:
    """Return whether a percentage run already has a trained checkpoint."""
    return (output_dir / "train_summary.json").exists() and (output_dir / "best").is_dir()


def has_all_metrics(output_dir: Path, splits: list[str]) -> bool:
    """Return whether all requested evaluation metrics already exist."""
    return all((output_dir / split / "metrics.json").exists() for split in splits)


def run_percent(percent: int, args: argparse.Namespace) -> None:
    """Train, infer, and evaluate one data-percentage condition."""
    dataset_dir = args.data_root / f"p{percent}"
    output_root = args.output_root / f"p{percent}"
    output_dir = output_root / args.model
    if has_all_metrics(output_dir, args.test_splits):
        print(f"Skipping p{percent}: all requested metrics exist.", flush=True)
        return
    cfg = train_config(args.model, dataset_dir, output_dir)
    config_path = output_dir / "train_config.json"
    write_json(config_path, cfg)
    if not has_completed_training(output_dir):
        run_command(command_prefix() + ["src/models/train_seq2seq.py", "--config", str(config_path)])
    else:
        print(f"Skipping p{percent} training: existing best checkpoint found.", flush=True)

    checkpoint = output_dir / "best"
    for split in args.test_splits:
        split_out = output_dir / split
        if (split_out / "metrics.json").exists():
            print(f"Skipping p{percent}/{split}: metrics exist.", flush=True)
            continue
        predictions = split_out / "predictions.jsonl"
        dataset_path = dataset_dir / "jsonl" / f"{split}.jsonl"
        run_command(
            command_prefix()
            + [
                "src/models/infer_seq2seq.py",
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                str(dataset_path),
                "--out",
                str(predictions),
                "--batch-size",
                "32",
                "--max-source-length",
                "160",
                "--max-new-tokens",
                "96",
            ]
        )
        run_command(
            command_prefix()
            + [
                "src/eval/evaluate_predictions.py",
                "--predictions",
                str(predictions),
                "--dataset",
                str(dataset_path),
                "--metrics-out",
                str(split_out / "metrics.json"),
                "--predictions-out",
                str(split_out / "evaluated_predictions.jsonl"),
                "--failure-breakdown-out",
                str(split_out / "failure_breakdown.json"),
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data-scaling experiments.")
    parser.add_argument("--data-root", type=Path, default=Path("data/generated/phase2_scaling/data_scaling"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase2_scaling/data_scaling"))
    parser.add_argument("--model", default="flan-t5-base")
    parser.add_argument("--percents", nargs="*", type=int, default=[1, 5, 10, 25, 50])
    parser.add_argument("--test-splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for percent in args.percents:
        run_percent(percent, args)


if __name__ == "__main__":
    main()

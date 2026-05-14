#!/usr/bin/env python3
"""Run Phase 2 supervised fine-tuning method ablations.

Pipeline stage: orchestration for comparing SFT target formats and training variants.
Inputs: method-specific datasets, model alias, and output-root settings.
Outputs: configs, checkpoints, predictions, and metrics by method.
Typical caller: scripts/run_phase2_sft_methods.sh and related launchers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import write_json
from src.models.model_registry import model_path
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS


DEFAULT_METHODS = [
    "coordinate_trace",
    "multi_template",
    "executor_feedback",
    "react_trace_arrow",
    "react_trace_compact",
    "react_trace_minimal",
    "react_trace_compact_sizegen",
]
TRACE_METHODS = {
    "coordinate_trace",
    "react_trace_arrow",
    "react_trace_compact",
    "react_trace_minimal",
    "react_trace_compact_sizegen",
}


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


def train_config(method: str, model_name: str, dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create the training config for one SFT method variant."""
    cfg: dict[str, Any] = {
        "model_name_or_path": str(model_path(model_name)),
        "output_dir": str(output_dir),
        "train_path": str(dataset_dir / "jsonl" / "train.jsonl"),
        "eval_path": str(dataset_dir / "jsonl" / "val.jsonl"),
        "max_source_length": 192,
        "max_target_length": 96,
        "num_train_epochs": 20,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.03,
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 3,
        "early_stopping_patience": 8,
        "eval_max_new_tokens": 96,
        "num_beams": 1,
        "seed": 1,
        "fp16": False,
        "bf16": model_name.startswith("flan-t5"),
    }
    if method == "coordinate_trace":
        cfg.update({"max_target_length": 256, "eval_max_new_tokens": 256, "selection_metric": "executor_success"})
    if method == "react_trace_arrow":
        cfg.update(
            {
                "max_target_length": 192,
                "eval_max_new_tokens": 192,
                "num_train_epochs": 30,
                "early_stopping_patience": 10,
                "selection_metric": "executor_success",
            }
        )
    if method == "react_trace_compact":
        cfg.update(
            {
                "max_target_length": 160,
                "eval_max_new_tokens": 160,
                "num_train_epochs": 30,
                "early_stopping_patience": 10,
                "selection_metric": "executor_success",
            }
        )
    if method == "react_trace_compact_sizegen":
        cfg.update(
            {
                "max_target_length": 256,
                "eval_max_new_tokens": 256,
                "num_train_epochs": 30,
                "early_stopping_patience": 10,
                "selection_metric": "executor_success",
            }
        )
    if method == "react_trace_minimal":
        cfg.update(
            {
                "max_target_length": 96,
                "eval_max_new_tokens": 96,
                "num_train_epochs": 30,
                "early_stopping_patience": 10,
                "selection_metric": "executor_success",
            }
        )
    return cfg


def has_completed_training(output_dir: Path) -> bool:
    """Return whether a method run already has a completed checkpoint."""
    return (output_dir / "train_summary.json").exists() and (output_dir / "best").is_dir()


def has_all_metrics(output_dir: Path, splits: list[str]) -> bool:
    """Return whether all requested method metrics already exist."""
    return all((output_dir / split / "metrics.json").exists() for split in splits)


def prepare_config(method: str, model_name: str, dataset_dir: Path, output_dir: Path) -> Path:
    """Write and return the training config path for one method run."""
    cfg = train_config(method, model_name, dataset_dir, output_dir)
    config_path = output_dir / "train_config.json"
    write_json(config_path, cfg)
    return config_path


def run_method(
    method: str,
    model_name: str,
    data_root: Path,
    output_root: Path,
    test_splits: list[str],
    allow_cpu: bool,
) -> None:
    """Train, infer, and evaluate one SFT method."""
    dataset_dir = data_root / method
    output_dir = output_root / method / model_name
    if has_all_metrics(output_dir, test_splits):
        print(f"Skipping {method}/{model_name}: all requested split metrics already exist.", flush=True)
        return

    compute_command = command_prefix() + ["src/models/check_compute.py", "--report-out", str(output_root / method / "compute_check.json")]
    if not allow_cpu:
        compute_command.append("--require-free-gpu")
    run_command(compute_command)

    config_path = prepare_config(method, model_name, dataset_dir, output_dir)
    if has_completed_training(output_dir):
        print(f"Skipping {method}/{model_name} training: existing best checkpoint found.", flush=True)
    else:
        run_command(command_prefix() + ["src/models/train_seq2seq.py", "--config", str(config_path)])

    checkpoint = output_dir / "best"
    for split in test_splits:
        split_out = output_dir / split
        if (split_out / "metrics.json").exists():
            print(f"Skipping {method}/{model_name}/{split}: metrics.json already exists.", flush=True)
            continue
        predictions_path = split_out / "predictions.jsonl"
        infer_command = command_prefix() + [
            "src/models/infer_seq2seq.py",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset_dir / "jsonl" / f"{split}.jsonl"),
            "--out",
            str(predictions_path),
            "--batch-size",
            "32",
            "--max-source-length",
            "192",
            "--max-new-tokens",
            "256"
            if method == "coordinate_trace"
            else "192"
            if method == "react_trace_arrow"
            else "160"
            if method == "react_trace_compact"
            else "256"
            if method == "react_trace_compact_sizegen"
            else "96",
        ]
        if method in TRACE_METHODS:
            infer_command.append("--extract-actions")
        run_command(infer_command)
        run_command(
            command_prefix()
            + [
                "src/eval/evaluate_predictions.py",
                "--predictions",
                str(predictions_path),
                "--dataset",
                str(dataset_dir / "jsonl" / f"{split}.jsonl"),
                "--metrics-out",
                str(split_out / "metrics.json"),
                "--predictions-out",
                str(split_out / "evaluated_predictions.jsonl"),
                "--failure-breakdown-out",
                str(split_out / "failure_breakdown.json"),
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 SFT-method ablations.")
    parser.add_argument("--data-root", type=Path, default=Path("data/generated/phase2_sft_method"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase2_sft_method"))
    parser.add_argument("--model", default="flan-t5-base")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--test-splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({"methods": args.methods, "model": args.model, "test_splits": args.test_splits}, indent=2))
    for method in args.methods:
        run_method(method, args.model, args.data_root, args.output_root, args.test_splits, args.allow_cpu)


if __name__ == "__main__":
    main()

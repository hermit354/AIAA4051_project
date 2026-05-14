#!/usr/bin/env python3
"""Run Phase 2 input-representation fine-tuning ablations.

Pipeline stage: orchestration for comparing natural, structured, grid, and hybrid inputs.
Inputs: converted input-format datasets, model alias, and output-root settings.
Outputs: configs, checkpoints, predictions, and metrics by input format.
Typical caller: scripts/run_phase2_input_format.sh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import write_json
from src.models.model_registry import model_path
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS, command_prefix, run_command


DEFAULT_FORMATS = ["structured", "grid_matrix", "hybrid"]


def train_config(model_name: str, dataset_root: Path, output_dir: Path, smoke: bool) -> dict[str, Any]:
    """Create a training config for one input-format ablation."""
    cfg: dict[str, Any] = {
        "model_name_or_path": str(model_path(model_name)),
        "output_dir": str(output_dir),
        "train_path": str(dataset_root / "jsonl" / "train.jsonl"),
        "eval_path": str(dataset_root / "jsonl" / "val.jsonl"),
        "max_source_length": 256,
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
        "fp16": not model_name.startswith("flan-t5"),
        "bf16": model_name.startswith("flan-t5"),
    }
    if smoke:
        cfg.update(
            {
                "num_train_epochs": 1,
                "max_steps": 2,
                "max_train_samples": 16,
                "max_eval_samples": 8,
                "eval_steps": 1,
                "save_steps": 1,
                "fp16": False,
                "bf16": False,
            }
        )
    return cfg


def ensure_compute(allow_cpu: bool, output_root: Path) -> None:
    """Run the compute preflight check for this experiment root."""
    command = command_prefix() + [
        "src/models/check_compute.py",
        "--report-out",
        str(output_root / "compute_check.json"),
    ]
    if not allow_cpu:
        command.append("--require-free-gpu")
    run_command(command)


def format_has_all_metrics(format_dir: Path, splits: list[str]) -> bool:
    """Return whether every requested split has already been evaluated."""
    return all((format_dir / split / "metrics.json").exists() for split in splits)


def format_has_completed_training(format_dir: Path) -> bool:
    """Return whether the format-specific model checkpoint already exists."""
    return (format_dir / "train_summary.json").exists() and (format_dir / "best").is_dir()


def run_pipeline(args: argparse.Namespace) -> None:
    """Run training, inference, and evaluation for each input format."""
    ensure_compute(args.allow_cpu, args.output_root)
    for input_format in args.formats:
        dataset_root = args.dataset_root / input_format
        format_dir = args.output_root / args.model / input_format
        if format_has_all_metrics(format_dir, args.test_splits):
            print(f"Skipping {input_format}: all requested split metrics already exist.", flush=True)
            continue

        config_path = format_dir / "train_config.json"
        write_json(config_path, train_config(args.model, dataset_root, format_dir, args.smoke))
        if format_has_completed_training(format_dir):
            print(f"Skipping {input_format} training: train_summary.json and best checkpoint already exist.", flush=True)
        else:
            run_command(command_prefix() + ["src/models/train_seq2seq.py", "--config", str(config_path)])

        checkpoint = format_dir / "best"
        for split in args.test_splits:
            split_out = format_dir / split
            if (split_out / "metrics.json").exists():
                print(f"Skipping {input_format}/{split}: metrics.json already exists.", flush=True)
                continue
            predictions_path = split_out / "predictions.jsonl"
            dataset_path = dataset_root / "jsonl" / f"{split}.jsonl"
            run_command(
                command_prefix()
                + [
                    "src/models/infer_seq2seq.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset",
                    str(dataset_path),
                    "--out",
                    str(predictions_path),
                    "--batch-size",
                    "16",
                    "--max-source-length",
                    "256",
                    "--max-new-tokens",
                    "96",
                ]
                + (["--max-samples", str(args.infer_max_samples)] if args.infer_max_samples is not None else [])
            )
            run_command(
                command_prefix()
                + [
                    "src/eval/evaluate_predictions.py",
                    "--predictions",
                    str(predictions_path),
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
    parser = argparse.ArgumentParser(description="Run Phase 2 input-format ablations.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/generated/phase2_input_format"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase2_input_format"))
    parser.add_argument("--model", default="flan-t5-base")
    parser.add_argument("--formats", nargs="*", default=DEFAULT_FORMATS)
    parser.add_argument("--test-splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--infer-max-samples", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        for input_format in args.formats:
            format_dir = args.output_root / args.model / input_format
            write_json(format_dir / "train_config.json", train_config(args.model, args.dataset_root / input_format, format_dir, args.smoke))
        print(json.dumps({"prepared": args.formats, "output_root": str(args.output_root)}, indent=2))
        return
    if args.infer_max_samples is None and args.smoke:
        args.infer_max_samples = 16
    run_pipeline(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and optionally run Phase 1 seq2seq baseline experiments.

Pipeline stage: orchestration for baseline training and evaluation.
Inputs: generated Phase 0 datasets, local model aliases, and output-root settings.
Outputs: training config JSON files, checkpoints, predictions, and metrics.
Typical caller: scripts/run_phase1_baselines_gpu5.sh.
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
from src.models.model_registry import model_names, model_path


DEFAULT_TEST_SPLITS = [
    "test_unseen_placement",
    "test_unseen_environment",
    "ood_size_5x5",
    "ood_size_7x7",
    "ood_size_8x8",
    "ood_size_9x9",
    "ood_size_10x10",
    "ood_dense_6x6",
    "ood_dense_5x5",
    "ood_dense_7x7",
    "ood_aspect_10x5",
    "ood_aspect_6x9",
]


def command_prefix() -> list[str]:
    """Return the Python command used by subprocess-based experiment runners.

    The Conda location and environment name are configurable so the same
    launcher can run on different machines.
    """
    conda_base = Path(os.environ.get("CONDA_BASE", str(Path.home() / "miniforge3")))
    conda_bin = Path(os.environ.get("CONDA_BIN", str(conda_base / "bin" / "conda")))
    env_name = os.environ.get("ENV_NAME", "NLP")
    return [str(conda_bin), "run", "-n", env_name, "python"]


def train_config(model_name: str, dataset_dir: Path, output_root: Path, smoke: bool) -> dict[str, Any]:
    """Create a reproducible training config for one baseline model."""
    batch_by_model = {
        "t5-small": 32,
        "flan-t5-small": 32,
        "t5-base": 16,
        "flan-t5-base": 16,
        "bart-base": 16,
    }
    cfg: dict[str, Any] = {
        "model_name_or_path": str(model_path(model_name)),
        "output_dir": str(output_root / model_name),
        "train_path": str(dataset_dir / "jsonl" / "train.jsonl"),
        "eval_path": str(dataset_dir / "jsonl" / "val.jsonl"),
        "max_source_length": 160,
        "max_target_length": 96,
        "num_train_epochs": 20,
        "per_device_train_batch_size": batch_by_model.get(model_name, 16),
        "per_device_eval_batch_size": batch_by_model.get(model_name, 16),
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
        "fp16": True,
        "bf16": False,
    }
    if model_name.startswith("flan-t5"):
        cfg["fp16"] = False
        cfg["bf16"] = True
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
            }
        )
    return cfg


def prepare_configs(models: list[str], dataset_dir: Path, output_root: Path, smoke: bool) -> list[Path]:
    """Write train_config.json files for the requested baseline models."""
    paths = []
    for model_name in models:
        cfg = train_config(model_name, dataset_dir, output_root, smoke)
        config_path = output_root / model_name / "train_config.json"
        write_json(config_path, cfg)
        paths.append(config_path)
    return paths


def run_command(command: list[str]) -> None:
    """Print and run one subprocess command with failure propagation."""
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def model_has_all_metrics(output_root: Path, model_name: str, test_splits: list[str]) -> bool:
    """Return whether all requested test metrics already exist."""
    return all((output_root / model_name / split / "metrics.json").exists() for split in test_splits)


def model_has_completed_training(output_root: Path, model_name: str) -> bool:
    """Return whether a baseline model already has a completed checkpoint."""
    model_dir = output_root / model_name
    return (model_dir / "train_summary.json").exists() and (model_dir / "best").is_dir()


def ensure_compute(allow_cpu: bool) -> None:
    """Run the GPU preflight check unless CPU fallback is explicitly allowed."""
    command = command_prefix() + ["src/models/check_compute.py", "--report-out", "outputs/compute_check.json"]
    if not allow_cpu:
        command.append("--require-free-gpu")
    run_command(command)


def run_pipeline(
    models: list[str],
    dataset_dir: Path,
    output_root: Path,
    test_splits: list[str],
    smoke: bool,
    allow_cpu: bool,
    infer_max_samples: int | None,
) -> None:
    """Train, infer, and evaluate all requested Phase 1 baseline models."""
    ensure_compute(allow_cpu)
    config_paths = prepare_configs(models, dataset_dir, output_root, smoke)
    for model_name, config_path in zip(models, config_paths):
        if model_has_all_metrics(output_root, model_name, test_splits):
            print(f"Skipping {model_name}: all requested split metrics already exist.", flush=True)
            continue
        if model_has_completed_training(output_root, model_name):
            print(f"Skipping {model_name} training: train_summary.json and best checkpoint already exist.", flush=True)
        else:
            run_command(command_prefix() + ["src/models/train_seq2seq.py", "--config", str(config_path)])
        checkpoint = output_root / model_name / "best"
        for split in test_splits:
            dataset_path = dataset_dir / "jsonl" / f"{split}.jsonl"
            split_out = output_root / model_name / split
            predictions_path = split_out / "predictions.jsonl"
            if (split_out / "metrics.json").exists():
                print(f"Skipping {model_name}/{split}: metrics.json already exists.", flush=True)
                continue
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
                    "--max-new-tokens",
                    "96",
                ]
                + (["--max-samples", str(infer_max_samples)] if infer_max_samples is not None else [])
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
    parser = argparse.ArgumentParser(description="Prepare or run Phase 1 seq2seq baselines.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/generated/phase0"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase1_baselines"))
    parser.add_argument("--models", nargs="*", default=model_names())
    parser.add_argument("--test-splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--infer-max-samples", type=int)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow training without a fully free GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        config_paths = prepare_configs(args.models, args.dataset_dir, args.output_root, args.smoke)
        print(json.dumps({"configs": [str(path) for path in config_paths]}, indent=2))
        return
    infer_max_samples = args.infer_max_samples
    if infer_max_samples is None and args.smoke:
        infer_max_samples = 16
    run_pipeline(args.models, args.dataset_dir, args.output_root, args.test_splits, args.smoke, args.allow_cpu, infer_max_samples)


if __name__ == "__main__":
    main()

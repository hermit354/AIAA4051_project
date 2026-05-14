#!/usr/bin/env python3
"""Generate compact ReAct-style trace data for size generalization.

Pipeline stage: dataset construction for trace-supervised Phase 2 experiments.
Inputs: generated environments, sampled start-goal pairs, and trace-format settings.
Outputs: JSON/JSONL trace datasets and manifests.
Typical caller: scripts/run_phase2_sizegen_trace_compact.sh.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_phase0 import generate_environments, generate_pairs, make_example
from src.data.make_sft_method_datasets import convert_react_trace
from src.data.ppnl_io import write_json, write_jsonl


TRAIN_SIZES = (5, 6, 7, 10)
TEST_SIZES = (8, 9, 12, 14, 16)
ENV_COUNTS = {1: 7, 2: 40, 3: 40, 4: 40, 5: 40}
TRAIN_PAIRS_PER_ENV = 24
VAL_PAIRS_PER_ENV = 3
TEST_PAIRS_PER_ENV = 30


def generate_size_rows(
    size: int,
    split: str,
    env_counts: dict[int, int],
    pairs_per_env: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    envs = generate_environments(size, size, env_counts, rng, f"sizegen_{split}")
    rows = []
    for env in envs:
        pairs = generate_pairs(env, pairs_per_env, rng)
        for pair_index, (start, goal, path) in enumerate(pairs):
            rows.append(make_example(env, split, pair_index, start, goal, path, is_dense=False))
    return rows, envs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, int] = {}
    by_obs: dict[str, int] = {}
    path_lengths = []
    for row in rows:
        by_shape[row["grid_shape"]] = by_shape.get(row["grid_shape"], 0) + 1
        key = str(row["obstacle_count"])
        by_obs[key] = by_obs.get(key, 0) + 1
        path_lengths.append(int(row["shortest_path_length"]))
    return {
        "examples": len(rows),
        "by_shape": dict(sorted(by_shape.items())),
        "by_obstacle_count": dict(sorted(by_obs.items(), key=lambda item: int(item[0]))),
        "shortest_path_length": {
            "min": min(path_lengths),
            "max": max(path_lengths),
            "mean": sum(path_lengths) / len(path_lengths),
        },
    }


def write_split(output_dir: Path, split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = convert_react_trace(rows, "react_trace_compact")
    write_jsonl(output_dir / "jsonl" / f"{split}.jsonl", converted)
    write_json(output_dir / "json" / f"{split}.json", converted)
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact trace size-generalization data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/phase2_size_generalization/react_trace_compact_sizegen"),
    )
    parser.add_argument("--seed", type=int, default=57)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    environments: list[dict[str, Any]] = []

    for size in TRAIN_SIZES:
        train_rows, train_envs = generate_size_rows(size, "train", ENV_COUNTS, TRAIN_PAIRS_PER_ENV, rng)
        val_rows, val_envs = generate_size_rows(size, "val", ENV_COUNTS, VAL_PAIRS_PER_ENV, rng)
        splits["train"].extend(train_rows)
        splits["val"].extend(val_rows)
        environments.extend(train_envs)
        environments.extend(val_envs)

    for size in TEST_SIZES:
        split = f"test_size_{size}x{size}"
        rows, envs = generate_size_rows(size, split, ENV_COUNTS, TEST_PAIRS_PER_ENV, rng)
        splits[split] = rows
        environments.extend(envs)

    converted_splits = {split: write_split(args.output_dir, split, rows) for split, rows in splits.items()}
    write_json(args.output_dir / "environments.json", environments)
    manifest = {
        "seed": args.seed,
        "method": "react_trace_compact_sizegen",
        "target_format": "react_trace_compact",
        "train_sizes": list(TRAIN_SIZES),
        "test_sizes": list(TEST_SIZES),
        "env_counts_per_size": ENV_COUNTS,
        "train_pairs_per_env": TRAIN_PAIRS_PER_ENV,
        "val_pairs_per_env": VAL_PAIRS_PER_ENV,
        "test_pairs_per_env": TEST_PAIRS_PER_ENV,
        "splits": {split: summarize(rows) for split, rows in converted_splits.items()},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

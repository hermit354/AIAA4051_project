#!/usr/bin/env python3
"""Generate multi-scale SFT train and validation data.

Pipeline stage: dataset construction for Phase 2 supervised fine-tuning.
Inputs: base Phase 0 records, requested grid sizes, and generation seeds.
Outputs: JSON/JSONL splits and manifests for multi-scale seq2seq training.
Typical caller: scripts/run_phase2_multiscale_sft.sh before model training.
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
from src.data.ppnl_io import load_records, write_json, write_jsonl
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS


SIZES = (4, 5, 6)
TRAIN_ENV_COUNTS = {1: 14, 2: 80, 3: 80, 4: 80, 5: 80}
VAL_ENV_COUNTS = {1: 14, 2: 80, 3: 80, 4: 80, 5: 80}
TRAIN_PAIRS_PER_ENV = 16
VAL_PAIRS_PER_ENV = 2


def generate_size_split(
    rows: int,
    cols: int,
    env_counts: dict[int, int],
    pairs_per_env: int,
    split: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    envs = generate_environments(rows, cols, env_counts, rng, f"multiscale_{split}")
    examples = []
    for env in envs:
        pairs = generate_pairs(env, pairs_per_env, rng)
        for pair_index, (start, goal, path) in enumerate(pairs):
            examples.append(make_example(env, split, pair_index, start, goal, path, is_dense=False))
    return examples, envs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, int] = {}
    by_obs: dict[str, int] = {}
    for row in rows:
        by_shape[row["grid_shape"]] = by_shape.get(row["grid_shape"], 0) + 1
        key = str(row["obstacle_count"])
        by_obs[key] = by_obs.get(key, 0) + 1
    return {
        "examples": len(rows),
        "by_shape": dict(sorted(by_shape.items())),
        "by_obstacle_count": dict(sorted(by_obs.items(), key=lambda item: int(item[0]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-scale SFT train/validation data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/phase2_multiscale_sft"))
    parser.add_argument("--phase0-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    splits = {"train": [], "val": []}
    environments: list[dict[str, Any]] = []
    for size in SIZES:
        train_rows, train_envs = generate_size_split(
            size, size, TRAIN_ENV_COUNTS, TRAIN_PAIRS_PER_ENV, "train", rng
        )
        val_rows, val_envs = generate_size_split(size, size, VAL_ENV_COUNTS, VAL_PAIRS_PER_ENV, "val", rng)
        splits["train"].extend(train_rows)
        splits["val"].extend(val_rows)
        environments.extend(train_envs)
        environments.extend(val_envs)

    for split, rows in splits.items():
        write_jsonl(args.output_dir / "jsonl" / f"{split}.jsonl", rows)
        write_json(args.output_dir / "json" / f"{split}.json", rows)
    for split in DEFAULT_TEST_SPLITS:
        rows = load_records(args.phase0_dir / f"{split}.jsonl")
        write_jsonl(args.output_dir / "jsonl" / f"{split}.jsonl", rows)
        write_json(args.output_dir / "json" / f"{split}.json", rows)
        splits[split] = rows
    write_json(args.output_dir / "environments.json", environments)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "sizes": list(SIZES),
        "train_env_counts_per_size": TRAIN_ENV_COUNTS,
        "val_env_counts_per_size": VAL_ENV_COUNTS,
        "train_pairs_per_env": TRAIN_PAIRS_PER_ENV,
        "val_pairs_per_env": VAL_PAIRS_PER_ENV,
        "splits": {split: summarize(rows) for split, rows in splits.items()},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

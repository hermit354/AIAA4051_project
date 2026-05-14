#!/usr/bin/env python3
"""Create stratified training subsets for data-scaling experiments.

Pipeline stage: dataset preparation for Phase 2.5 scaling analysis.
Inputs: a full training split, evaluation splits, and requested percentage levels.
Outputs: smaller JSONL training subsets with copied validation/test splits and a manifest.
Typical caller: scripts/run_phase2_data_scaling.sh.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json, write_jsonl
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS


def stratified_subset(records: list[dict[str, Any]], percent: int, seed: int) -> list[dict[str, Any]]:
    if percent >= 100:
        return list(records)
    by_obstacles: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_obstacles[int(record["obstacle_count"])].append(record)

    rng = random.Random(seed + percent)
    selected: list[dict[str, Any]] = []
    for obstacle_count in sorted(by_obstacles):
        group = list(by_obstacles[obstacle_count])
        rng.shuffle(group)
        keep = max(1, round(len(group) * percent / 100))
        selected.extend(group[:keep])
    selected.sort(key=lambda row: str(row.get("example_id", row.get("id", ""))))
    return selected


def obstacle_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["obstacle_count"])] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data-scaling JSONL datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/generated/phase2_scaling/data_scaling"))
    parser.add_argument("--percents", nargs="*", type=int, default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--seed", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_records = load_records(args.source_dir / "train.jsonl")
    manifest: dict[str, Any] = {"source_dir": str(args.source_dir), "percents": {}, "seed": args.seed}
    for percent in args.percents:
        root = args.output_root / f"p{percent}" / "jsonl"
        subset = stratified_subset(train_records, percent, args.seed)
        write_jsonl(root / "train.jsonl", subset)
        for split in ["val", *DEFAULT_TEST_SPLITS]:
            rows = load_records(args.source_dir / f"{split}.jsonl")
            write_jsonl(root / f"{split}.jsonl", rows)
        manifest["percents"][str(percent)] = {
            "root": str(args.output_root / f"p{percent}"),
            "train_count": len(subset),
            "train_obstacle_counts": obstacle_counts(subset),
        }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

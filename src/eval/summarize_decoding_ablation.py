#!/usr/bin/env python3
"""Summarize decoding-ablation metrics by ID and OOD groups.

Pipeline stage: result aggregation after decoding experiments.
Inputs: per-split metrics.json files from decoding-ablation output directories.
Outputs: CSV tables for reports and plots.
Typical caller: figure/table generation workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.summarize_phase1 import GROUPS


def load_metrics(root: Path, strategy: str, split: str) -> dict[str, Any]:
    with (root / strategy / split / "metrics.json").open() as handle:
        return json.load(handle)


def weighted_average(items: list[dict[str, Any]], key: str) -> float:
    total = sum(item["total"] for item in items)
    return sum(item[key] * item["total"] for item in items) / total


def summarize_strategy(root: Path, model: str, strategy: str) -> list[dict[str, Any]]:
    rows = []
    for group, splits in GROUPS.items():
        metrics_paths = [root / strategy / split / "metrics.json" for split in splits]
        if not all(path.exists() for path in metrics_paths):
            continue
        metrics = [load_metrics(root, strategy, split) for split in splits]
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "group": group,
                "splits": "+".join(splits),
                "total": sum(item["total"] for item in metrics),
                "success_rate": weighted_average(metrics, "success_rate"),
                "feasibility": weighted_average(metrics, "feasibility"),
                "optimality": weighted_average(metrics, "optimality"),
                "exact_match": weighted_average(metrics, "exact_match"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize decoding-ablation metrics by OOD group.")
    parser.add_argument("--root", type=Path, default=Path("outputs/phase2_decoding/flan-t5-base_full"))
    parser.add_argument("--model", default="flan-t5-base")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["greedy", "beam4", "beam8", "token_constrained", "postprocess", "rerank_beam8", "state_aware"],
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_decoding/flan-t5-base_full/tables/decoding_group_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for strategy in args.strategies:
        rows.extend(summarize_strategy(args.root, args.model, strategy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"no complete strategy/group metrics found under {args.root}")
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

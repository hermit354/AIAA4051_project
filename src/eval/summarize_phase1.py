#!/usr/bin/env python3
"""Summarize Phase 1 baseline metrics with ID and OOD group breakdowns.

Pipeline stage: baseline result aggregation.
Inputs: model output roots containing train summaries and per-split metrics.
Outputs: grouped CSV tables used in reports and README excerpts.
Typical caller: baseline experiment scripts and manual report updates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GROUPS = {
    "id_unseen_placement": ["test_unseen_placement"],
    "id_unseen_environment": ["test_unseen_environment"],
    "ood_size_5x5": ["ood_size_5x5"],
    "ood_size_7x7": ["ood_size_7x7"],
    "ood_size_8x8": ["ood_size_8x8"],
    "ood_size_9x9": ["ood_size_9x9"],
    "ood_size_10x10": ["ood_size_10x10"],
    "ood_dense": ["ood_dense_5x5", "ood_dense_6x6", "ood_dense_7x7"],
    "ood_dense_5x5": ["ood_dense_5x5"],
    "ood_dense_6x6": ["ood_dense_6x6"],
    "ood_dense_7x7": ["ood_dense_7x7"],
    "ood_aspect": ["ood_aspect_10x5", "ood_aspect_6x9"],
    "ood_aspect_10x5": ["ood_aspect_10x5"],
    "ood_aspect_6x9": ["ood_aspect_6x9"],
}


def load_metrics(root: Path, model: str, split: str) -> dict[str, Any]:
    with (root / model / split / "metrics.json").open() as handle:
        return json.load(handle)


def weighted_average(items: list[dict[str, Any]], key: str) -> float:
    total = sum(item["total"] for item in items)
    return sum(item[key] * item["total"] for item in items) / total


def summarize_model(root: Path, model: str) -> list[dict[str, Any]]:
    with (root / model / "train_summary.json").open() as handle:
        train = json.load(handle)
    rows = []
    for group, splits in GROUPS.items():
        metrics = [load_metrics(root, model, split) for split in splits]
        rows.append(
            {
                "model": model,
                "group": group,
                "splits": "+".join(splits),
                "total": sum(item["total"] for item in metrics),
                "best_val_exact": train.get("best_exact_match"),
                "success_rate": weighted_average(metrics, "success_rate"),
                "feasibility": weighted_average(metrics, "feasibility"),
                "optimality": weighted_average(metrics, "optimality"),
                "exact_match": weighted_average(metrics, "exact_match"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 1 baseline metrics by OOD group.")
    parser.add_argument("--root", type=Path, default=Path("outputs/phase1_baselines_converged"))
    parser.add_argument("--models", nargs="*", default=["t5-small", "t5-base", "flan-t5-small", "flan-t5-base", "bart-base"])
    parser.add_argument("--out", type=Path, default=Path("outputs/phase1_baselines_converged/tables/phase1_group_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for model in args.models:
        if (args.root / model / "train_summary.json").exists():
            rows.extend(summarize_model(args.root, model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

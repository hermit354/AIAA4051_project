#!/usr/bin/env python3
"""Summarize input-format SFT metrics by ID and OOD groups.

Pipeline stage: result aggregation after input-representation experiments.
Inputs: metrics.json files for natural, structured, grid, or hybrid input formats.
Outputs: CSV summaries for comparison tables.
Typical caller: report-generation commands.
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def weighted_average(items: list[dict[str, Any]], key: str) -> float:
    total = sum(item["total"] for item in items)
    return sum(item[key] * item["total"] for item in items) / total


def summarize_format(root: Path, model: str, input_format: str) -> list[dict[str, Any]]:
    format_dir = root / model / input_format
    train_summary = load_json(format_dir / "train_summary.json") if (format_dir / "train_summary.json").exists() else {}
    rows = []
    for group, splits in GROUPS.items():
        paths = [format_dir / split / "metrics.json" for split in splits]
        if not all(path.exists() for path in paths):
            continue
        metrics = [load_json(path) for path in paths]
        rows.append(
            {
                "model": model,
                "input_format": input_format,
                "group": group,
                "splits": "+".join(splits),
                "total": sum(item["total"] for item in metrics),
                "best_val_exact": train_summary.get("best_exact_match"),
                "success_rate": weighted_average(metrics, "success_rate"),
                "feasibility": weighted_average(metrics, "feasibility"),
                "optimality": weighted_average(metrics, "optimality"),
                "exact_match": weighted_average(metrics, "exact_match"),
            }
        )
    return rows


def load_natural_baseline(path: Path, model: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model:
                continue
            row = dict(row)
            row["input_format"] = "natural"
            rows.append(
                {
                    "model": row["model"],
                    "input_format": row["input_format"],
                    "group": row["group"],
                    "splits": row["splits"],
                    "total": row["total"],
                    "best_val_exact": row.get("best_val_exact"),
                    "success_rate": row["success_rate"],
                    "feasibility": row["feasibility"],
                    "optimality": row["optimality"],
                    "exact_match": row["exact_match"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize input-format ablation metrics.")
    parser.add_argument("--root", type=Path, default=Path("outputs/phase2_input_format"))
    parser.add_argument("--model", default="flan-t5-base")
    parser.add_argument("--formats", nargs="*", default=["structured", "grid_matrix", "hybrid"])
    parser.add_argument("--natural-baseline", type=Path, default=Path("outputs/tables/table1_phase1_group_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_input_format/tables/input_format_group_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_natural_baseline(args.natural_baseline, args.model)
    for input_format in args.formats:
        rows.extend(summarize_format(args.root, args.model, input_format))
    if not rows:
        raise SystemExit("no input-format metrics found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

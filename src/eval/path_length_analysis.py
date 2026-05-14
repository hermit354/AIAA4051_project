#!/usr/bin/env python3
"""Summarize model performance by shortest-path length buckets.

Pipeline stage: post-evaluation diagnostic analysis.
Inputs: evaluated prediction rows with shortest-path and success fields.
Outputs: CSV summaries that connect path length to model performance.
Typical caller: manual analysis for report discussion and failure interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS


def bucket(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length <= 8:
        return "short"
    if length <= 14:
        return "medium"
    return "long"


def summarize_file(path: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_records(path):
        groups[bucket(row.get("shortest_path_length"))].append(row)
    summary = {}
    for name, rows in groups.items():
        total = len(rows)
        summary[name] = {
            "total": total,
            "success_rate": sum(bool(row.get("success")) for row in rows) / total,
            "feasibility": sum(bool(row.get("feasible")) for row in rows) / total,
            "optimality": sum(bool(row.get("optimal")) for row in rows) / total,
            "exact_match": sum(bool(row.get("exact_match")) for row in rows) / total,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path-length performance analysis.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_scaling/tables/path_length_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for label, run_dir_raw in args.run:
        run_dir = Path(run_dir_raw)
        for split in args.splits:
            path = run_dir / split / "evaluated_predictions.jsonl"
            if not path.exists():
                continue
            for bucket_name, metrics in summarize_file(path).items():
                rows.append({"label": label, "split": split, "path_length_bucket": bucket_name, **metrics})
    if not rows:
        raise SystemExit("no evaluated prediction files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

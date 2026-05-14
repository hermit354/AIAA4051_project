#!/usr/bin/env python3
"""Summarize stepwise rollout failures by path length, distance, and loop behavior.

Pipeline stage: post-evaluation diagnostics.
Inputs: evaluated prediction JSONL files produced by src/eval/evaluate_predictions.py.
Outputs: aggregate JSON summaries for interpreting why rollout variants fail.
Typical caller: manual analysis commands used during model and report iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json


def has_backtrack(trajectory: list[tuple[int, int]]) -> bool:
    return any(index >= 2 and trajectory[index] == trajectory[index - 2] for index in range(2, len(trajectory)))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize_split(path: Path) -> dict[str, Any]:
    rows = load_records(path)
    failures = [row for row in rows if not row.get("success")]
    loop_count = 0
    backtrack_count = 0
    repeated_states: list[float] = []
    final_distances: list[float] = []
    generated_lengths: list[float] = []
    shortest_lengths: list[float] = []
    ratios: list[float] = []
    progress: list[float] = []
    failure_types: dict[str, int] = {}

    for row in failures:
        trajectory = [tuple(item) for item in row.get("executed_trajectory", [])]
        loop_count += int(len(set(trajectory)) < len(trajectory))
        backtrack_count += int(has_backtrack(trajectory))
        repeated_states.append(float(len(trajectory) - len(set(trajectory))))
        if row.get("final_distance_to_goal") is not None:
            final_distances.append(float(row["final_distance_to_goal"]))
        if row.get("generated_path_length") is not None:
            generated_lengths.append(float(row["generated_path_length"]))
        if row.get("shortest_path_length") is not None:
            shortest_lengths.append(float(row["shortest_path_length"]))
        if row.get("path_length_ratio") is not None:
            ratios.append(float(row["path_length_ratio"]))
        if row.get("shortest_path_length") is not None and row.get("final_distance_to_goal") is not None:
            progress.append(float(row["shortest_path_length"]) - float(row["final_distance_to_goal"]))
        failure_type = str(row.get("failure_type", "UNKNOWN"))
        failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

    total_failures = len(failures)
    return {
        "total": len(rows),
        "failures": total_failures,
        "failure_rate": total_failures / len(rows) if rows else None,
        "failure_types": dict(sorted(failure_types.items())),
        "loop_failure_rate": loop_count / total_failures if total_failures else 0.0,
        "backtrack_failure_rate": backtrack_count / total_failures if total_failures else 0.0,
        "avg_repeated_states": mean(repeated_states),
        "avg_final_distance_to_goal": mean(final_distances),
        "median_final_distance_to_goal": median(final_distances),
        "avg_generated_path_length": mean(generated_lengths),
        "avg_shortest_path_length": mean(shortest_lengths),
        "avg_path_length_ratio": mean(ratios),
        "avg_shortest_progress": mean(progress),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stepwise evaluated failure files.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {
        split: summarize_split(args.root / split / "evaluated_predictions.jsonl")
        for split in args.splits
    }
    if args.out:
        write_json(args.out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze residual failures from top-k stepwise rollout outputs.

Pipeline stage: post-evaluation error analysis.
Inputs: evaluated prediction JSONL files with trajectories, failure types, and distance fields.
Outputs: JSON summaries and JSONL detail rows that describe loop, distance, and residual error patterns.
Typical caller: manual analysis commands or report-generation workflows after stepwise evaluation finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json, write_jsonl


def mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, preserving None for empty inputs."""
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    """Return the median, preserving None for empty inputs."""
    return float(statistics.median(values)) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    """Return a nearest-rank percentile for a numeric list."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def tuple_path(row: dict[str, Any]) -> list[tuple[int, int]]:
    """Convert an evaluated trajectory row into hashable coordinate tuples."""
    return [tuple(item) for item in row.get("executed_trajectory", [])]


def has_loop(trajectory: list[tuple[int, int]]) -> bool:
    """Return whether a trajectory revisits any grid cell."""
    return len(set(trajectory)) < len(trajectory)


def has_backtrack(trajectory: list[tuple[int, int]]) -> bool:
    """Return whether the trajectory immediately returns to a previous cell."""
    return any(index >= 2 and trajectory[index] == trajectory[index - 2] for index in range(2, len(trajectory)))


def obstacle_distance(cell: tuple[int, int], obstacles: set[tuple[int, int]]) -> int | None:
    """Return the Manhattan distance from a cell to the nearest obstacle."""
    if not obstacles:
        return None
    return min(abs(cell[0] - obs[0]) + abs(cell[1] - obs[1]) for obs in obstacles)


def near_obstacle(cell: tuple[int, int], obstacles: set[tuple[int, int]], radius: int) -> bool:
    """Return whether a cell is within `radius` of any obstacle."""
    distance = obstacle_distance(cell, obstacles)
    return distance is not None and distance <= radius


def local_oscillation_segments(trajectory: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """Identify repeated A-B-A-B oscillation spans in a trajectory."""
    segments: list[dict[str, Any]] = []
    index = 2
    while index < len(trajectory):
        if trajectory[index] != trajectory[index - 2]:
            index += 1
            continue
        start = index - 2
        a = trajectory[start]
        b = trajectory[start + 1]
        end = index
        while end + 1 < len(trajectory) and trajectory[end + 1] == trajectory[end - 1]:
            end += 1
        segments.append({"start": start, "end": end, "cells": [list(a), list(b)], "steps": end - start})
        index = end + 1
    return segments


def max_step_budget(row: dict[str, Any], length_factor: float, absolute_max_steps: int) -> int | None:
    """Reconstruct the rollout step budget used for a row."""
    shortest = row.get("shortest_path_length")
    if shortest is None:
        return None
    return max(1, int(math.ceil(min(absolute_max_steps, length_factor * int(shortest)))))


def summarize_failures(
    rows: list[dict[str, Any]],
    length_factor: float,
    absolute_max_steps: int,
    obstacle_radius: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate residual failure patterns and produce detailed diagnostics.

    The summary captures how often failures involve loops, immediate
    backtracking, obstacle proximity, or budget exhaustion. The details output
    keeps one compact diagnostic row per failed example for later inspection.
    """
    failures = [row for row in rows if not row.get("success")]
    failure_types = Counter(str(row.get("failure_type", "UNKNOWN")) for row in failures)

    final_distances: list[float] = []
    generated_lengths: list[float] = []
    budgets: list[float] = []
    shortest_lengths: list[float] = []
    repeated_states: list[float] = []
    min_obstacle_distances: list[float] = []
    final_obstacle_distances: list[float] = []

    loop_count = 0
    backtrack_count = 0
    oscillation_count = 0
    near_obstacle_oscillation_count = 0
    near_obstacle_any_count = 0
    max_budget_count = 0
    max_budget_with_loop_count = 0
    detailed: list[dict[str, Any]] = []

    for row in failures:
        trajectory = tuple_path(row)
        obstacles = {tuple(item) for item in row.get("obstacles", [])}
        budget = max_step_budget(row, length_factor, absolute_max_steps)
        generated_length = int(row.get("generated_path_length", 0))
        shortest = row.get("shortest_path_length")
        final_distance = row.get("final_distance_to_goal")
        loops = has_loop(trajectory)
        backtracks = has_backtrack(trajectory)
        oscillations = local_oscillation_segments(trajectory)
        near_oscillations = [
            segment
            for segment in oscillations
            if any(near_obstacle(tuple(cell), obstacles, obstacle_radius) for cell in map(tuple, segment["cells"]))
        ]
        min_obs_distance = min(
            (distance for distance in (obstacle_distance(cell, obstacles) for cell in trajectory) if distance is not None),
            default=None,
        )
        final_obs_distance = obstacle_distance(trajectory[-1], obstacles) if trajectory else None
        hit_budget = bool(budget is not None and generated_length >= budget and not row.get("success"))

        loop_count += int(loops)
        backtrack_count += int(backtracks)
        oscillation_count += int(bool(oscillations))
        near_obstacle_oscillation_count += int(bool(near_oscillations))
        near_obstacle_any_count += int(min_obs_distance is not None and min_obs_distance <= obstacle_radius)
        max_budget_count += int(hit_budget)
        max_budget_with_loop_count += int(hit_budget and loops)
        repeated_states.append(float(len(trajectory) - len(set(trajectory))))
        if final_distance is not None:
            final_distances.append(float(final_distance))
        if generated_length is not None:
            generated_lengths.append(float(generated_length))
        if budget is not None:
            budgets.append(float(budget))
        if shortest is not None:
            shortest_lengths.append(float(shortest))
        if min_obs_distance is not None:
            min_obstacle_distances.append(float(min_obs_distance))
        if final_obs_distance is not None:
            final_obstacle_distances.append(float(final_obs_distance))

        detailed.append(
            {
                "id": row.get("id"),
                "failure_type": row.get("failure_type"),
                "start": row.get("start"),
                "goal": row.get("goal"),
                "shortest_path_length": shortest,
                "generated_path_length": generated_length,
                "max_step_budget": budget,
                "hit_max_step_budget": hit_budget,
                "final_distance_to_goal": final_distance,
                "path_length_ratio": row.get("path_length_ratio"),
                "has_loop": loops,
                "has_backtrack": backtracks,
                "repeated_states": len(trajectory) - len(set(trajectory)),
                "oscillation_segments": oscillations,
                "near_obstacle_oscillation": bool(near_oscillations),
                "near_obstacle_oscillation_segments": near_oscillations,
                "min_distance_to_obstacle_along_path": min_obs_distance,
                "final_distance_to_nearest_obstacle": final_obs_distance,
                "obstacle_count": len(obstacles),
                "parsed_actions": row.get("parsed_actions", []),
                "executed_trajectory": row.get("executed_trajectory", []),
            }
        )

    total = len(rows)
    total_failures = len(failures)
    summary = {
        "total": total,
        "successes": total - total_failures,
        "failures": total_failures,
        "failure_rate": total_failures / total if total else None,
        "failure_types": dict(sorted(failure_types.items())),
        "final_distance_to_goal": {
            "mean": mean(final_distances),
            "median": median(final_distances),
            "p90": percentile(final_distances, 90),
            "max": max(final_distances) if final_distances else None,
            "histogram": dict(sorted(Counter(int(value) for value in final_distances).items())),
        },
        "loop": {
            "count": loop_count,
            "rate_among_failures": loop_count / total_failures if total_failures else 0.0,
            "backtrack_count": backtrack_count,
            "backtrack_rate_among_failures": backtrack_count / total_failures if total_failures else 0.0,
            "avg_repeated_states": mean(repeated_states),
        },
        "local_oscillation": {
            "count": oscillation_count,
            "rate_among_failures": oscillation_count / total_failures if total_failures else 0.0,
            "near_obstacle_count": near_obstacle_oscillation_count,
            "near_obstacle_rate_among_failures": near_obstacle_oscillation_count / total_failures if total_failures else 0.0,
            "near_obstacle_radius": obstacle_radius,
            "any_path_near_obstacle_count": near_obstacle_any_count,
            "any_path_near_obstacle_rate_among_failures": near_obstacle_any_count / total_failures if total_failures else 0.0,
            "min_distance_to_obstacle_along_path_mean": mean(min_obstacle_distances),
            "final_distance_to_nearest_obstacle_mean": mean(final_obstacle_distances),
            "min_distance_to_obstacle_histogram": dict(sorted(Counter(int(value) for value in min_obstacle_distances).items())),
        },
        "max_step_budget": {
            "length_factor": length_factor,
            "absolute_max_steps": absolute_max_steps,
            "count": max_budget_count,
            "rate_among_failures": max_budget_count / total_failures if total_failures else 0.0,
            "with_loop_count": max_budget_with_loop_count,
            "with_loop_rate_among_budget_hits": max_budget_with_loop_count / max_budget_count if max_budget_count else 0.0,
            "avg_generated_path_length": mean(generated_lengths),
            "avg_budget": mean(budgets),
            "avg_shortest_path_length": mean(shortest_lengths),
        },
    }
    detailed.sort(
        key=lambda row: (
            not row["hit_max_step_budget"],
            not row["has_loop"],
            -(row["final_distance_to_goal"] or 0),
            str(row["id"]),
        )
    )
    return summary, detailed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze residual top-k stepwise rollout failures.")
    parser.add_argument("--evaluated", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--details-out", required=True, type=Path)
    parser.add_argument("--length-factor", type=float, default=3.0)
    parser.add_argument("--absolute-max-steps", type=int, default=96)
    parser.add_argument("--obstacle-radius", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_records(args.evaluated)
    summary, details = summarize_failures(rows, args.length_factor, args.absolute_max_steps, args.obstacle_radius)
    write_json(args.summary_out, summary)
    write_jsonl(args.details_out, details)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

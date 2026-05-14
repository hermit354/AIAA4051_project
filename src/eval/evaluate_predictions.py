#!/usr/bin/env python3
"""Evaluate grid path-planning predictions by executing actions in the grid.

Pipeline stage: central evaluation step for all direct and stepwise model outputs.
Inputs: prediction JSON/JSONL/CSV files and optional aligned datasets.
Outputs: aggregate metrics, per-example execution traces, and failure breakdowns.
Typical caller: training runners, rollout launchers, sanity checks, and report scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json, write_jsonl


ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FAILURE_PRIORITY = (
    "INVALID_TOKEN",
    "OUT_OF_BOUNDS",
    "HIT_OBSTACLE",
    "PREMATURE_STOP",
    "LOOP_TOO_LONG",
    "NON_OPTIMAL_SUCCESS",
    "SUCCESS_OPTIMAL",
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _as_text(value[0]) if value else ""
    return str(value)


def tokenize_actions(value: Any) -> tuple[list[str], str]:
    """Normalize a model output into action tokens and canonical text."""
    text = _as_text(value).strip().lower()
    if not text:
        return [], text
    text = text.replace(",", " ")
    return [token for token in text.split() if token], " ".join(text.split())


def validate_world(world: Any) -> list[list[int]]:
    if not isinstance(world, list) or not world or not all(isinstance(row, list) for row in world):
        raise ValueError("world must be a non-empty 2D list")
    width = len(world[0])
    if width == 0 or any(len(row) != width for row in world):
        raise ValueError("world must be rectangular")
    return [[int(cell) for cell in row] for row in world]


def find_marker(world: list[list[int]], marker: int) -> tuple[int, int]:
    matches = []
    for row_index, row in enumerate(world):
        for col_index, cell in enumerate(row):
            if cell == marker:
                matches.append((row_index, col_index))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one marker {marker}, found {len(matches)}")
    return matches[0]


def shortest_path_length(world: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    """Return the shortest executable path length in the grid.

    The evaluator uses BFS because grid moves have uniform cost. This makes
    optimality a behavior-level metric rather than a text-match metric.
    """
    if start == goal:
        return 0
    rows, cols = len(world), len(world[0])
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), distance = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if world[nxt[0]][nxt[1]] == 1 or nxt in visited:
                continue
            if nxt == goal:
                return distance + 1
            visited.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def distance_to_goal(world: list[list[int]], current: tuple[int, int], goal: tuple[int, int]) -> int | None:
    return shortest_path_length(world, current, goal)


def difficulty_metrics(world: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> dict[str, Any]:
    rows, cols = len(world), len(world[0])
    obstacles = sum(cell == 1 for row in world for cell in row)
    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    shortest = shortest_path_length(world, start, goal)
    return {
        "grid_rows": rows,
        "grid_cols": cols,
        "obstacle_count": obstacles,
        "obstacle_density": obstacles / (rows * cols),
        "manhattan_distance": manhattan,
        "shortest_path_length": shortest,
        "detour_ratio": None if shortest is None or manhattan == 0 else shortest / manhattan,
    }


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize compatible dataset/prediction schemas into evaluator fields."""
    world = validate_world(record["world"])
    start = tuple(record.get("start") or find_marker(world, 2))
    if "goal" in record:
        goal = tuple(record["goal"])
    elif "goals" in record and record["goals"]:
        goal = tuple(record["goals"][0])
    else:
        goal = find_marker(world, 3)
    return {
        **record,
        "id": record.get("id", record.get("example_id", index)),
        "world": world,
        "start": list(start),
        "goal": list(goal),
        "input_grid": record.get("grid_matrix_input", record.get("english", record.get("nl_description", ""))),
        "ground_truth": record.get("ground_truth", record.get("agent_as_a_point", record.get("target", ""))),
        "model_output": record.get("model_output", record.get("generated", record.get("prediction", ""))),
    }


def execute_plan(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Execute one predicted action sequence and return detailed metrics.

    Args:
        record: A merged dataset/prediction record with `world`, start/goal
            information, ground-truth text, and model-output text.
        index: Fallback integer id when the record has no explicit id.

    Returns:
        Feasibility, success, optimality, failure type, trajectory, and
        difficulty metrics for one model prediction.
    """
    record = normalize_record(record, index)
    world = record["world"]
    rows, cols = len(world), len(world[0])
    start = tuple(record["start"])
    goal = tuple(record["goal"])
    shortest = record.get("shortest_path_length")
    if shortest is None:
        shortest = shortest_path_length(world, start, goal)

    parsed_actions, normalized_prediction = tokenize_actions(record["model_output"])
    _, normalized_gold = tokenize_actions(record["ground_truth"])
    current = start
    trajectory = [list(current)]
    failure_type: str | None = None
    first_error_step: int | None = None
    trace = []

    max_reasonable_length = None if shortest is None else 3 * int(shortest)
    for step_index, action in enumerate(parsed_actions):
        before = current
        # Failure priority is executor-first: malformed text or illegal
        # movement should be diagnosed before final success checks.
        if action not in ACTION_DELTAS:
            failure_type = "INVALID_TOKEN"
            first_error_step = step_index
            trace.append({"step": step_index, "action": action, "from": list(before), "to": list(before), "status": failure_type})
            break

        dr, dc = ACTION_DELTAS[action]
        after = (before[0] + dr, before[1] + dc)
        if not (0 <= after[0] < rows and 0 <= after[1] < cols):
            failure_type = "OUT_OF_BOUNDS"
            first_error_step = step_index
            trace.append({"step": step_index, "action": action, "from": list(before), "to": list(after), "status": failure_type})
            break
        if world[after[0]][after[1]] == 1:
            failure_type = "HIT_OBSTACLE"
            first_error_step = step_index
            trace.append({"step": step_index, "action": action, "from": list(before), "to": list(after), "status": failure_type})
            break

        current = after
        trajectory.append(list(current))
        trace.append({"step": step_index, "action": action, "from": list(before), "to": list(after), "status": "OK"})

        if max_reasonable_length is not None and len(trajectory) - 1 > max_reasonable_length and current != goal:
            failure_type = "LOOP_TOO_LONG"
            first_error_step = step_index
            break

    feasible = failure_type is None
    success = feasible and current == goal
    generated_path_length = len(parsed_actions)
    optimal = bool(success and shortest is not None and generated_path_length == int(shortest))

    if failure_type is None:
        if not success:
            failure_type = "PREMATURE_STOP"
        elif not optimal:
            failure_type = "NON_OPTIMAL_SUCCESS"
        else:
            failure_type = "SUCCESS_OPTIMAL"

    final_distance = distance_to_goal(world, current, goal) if feasible else None
    path_length_ratio = None if shortest in (None, 0) else generated_path_length / int(shortest)

    return {
        "id": record["id"],
        "input_grid": record["input_grid"],
        "start": record["start"],
        "goal": record["goal"],
        "obstacles": record.get("obstacles", []),
        "shortest_path_length": shortest,
        "model_output": record["model_output"],
        "parsed_actions": parsed_actions,
        "executed_trajectory": trajectory,
        "feasible": feasible,
        "success": success,
        "optimal": optimal,
        "failure_type": failure_type,
        "first_error_step": first_error_step,
        "final_distance_to_goal": final_distance,
        "generated_path_length": generated_path_length,
        "path_length_ratio": path_length_ratio,
        "exact_match": normalized_prediction == normalized_gold,
        "difficulty": difficulty_metrics(world, start, goal),
        "trace": trace,
    }


def load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    return load_records(path)


def merge_prediction_and_dataset(
    predictions: list[dict[str, Any]], dataset: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if dataset is not None and len(predictions) != len(dataset):
        by_id = {str(row.get("id", row.get("example_id", index))): row for index, row in enumerate(dataset)}
        if all(str(prediction.get("id")) in by_id for prediction in predictions):
            dataset = [by_id[str(prediction.get("id"))] for prediction in predictions]
        elif len(predictions) < len(dataset):
            dataset = dataset[: len(predictions)]
        else:
            raise ValueError(f"prediction count {len(predictions)} != dataset count {len(dataset)}")
    merged = []
    for index, prediction in enumerate(predictions):
        base = dict(dataset[index]) if dataset is not None else {}
        row = {**base, **prediction}
        if "model_output" not in row and "generated" in row:
            row["model_output"] = row["generated"]
        if "model_output" not in row and "prediction" in row:
            row["model_output"] = row["prediction"]
        if "world" not in row:
            raise ValueError(f"record {index} has no world field; pass --dataset or include world in predictions")
        merged.append(row)
    return merged


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        raise ValueError("no records to evaluate")
    failure_counts = Counter(row["failure_type"] for row in results)
    feasible_distances = [row["final_distance_to_goal"] for row in results if row["final_distance_to_goal"] is not None]
    ratios = [row["path_length_ratio"] for row in results if row["path_length_ratio"] is not None]
    return {
        "total": total,
        "success_rate": sum(1 for row in results if row["success"]) / total,
        "feasibility": sum(1 for row in results if row["feasible"]) / total,
        "optimality": sum(1 for row in results if row["optimal"]) / total,
        "exact_match": sum(1 for row in results if row["exact_match"]) / total,
        "avg_final_distance_to_goal_feasible": sum(feasible_distances) / len(feasible_distances) if feasible_distances else None,
        "avg_path_length_ratio": sum(ratios) / len(ratios) if ratios else None,
        "failure_counts": {key: failure_counts.get(key, 0) for key in FAILURE_PRIORITY if failure_counts.get(key, 0)},
    }


def evaluate_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = [execute_plan(record, index) for index, record in enumerate(records)]
    return summarize(results), results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-goal grid path-planning predictions.")
    parser.add_argument("--predictions", required=True, type=Path, help="JSON, JSONL, or CSV prediction file.")
    parser.add_argument("--dataset", type=Path, help="Optional aligned dataset JSON/JSONL.")
    parser.add_argument("--metrics-out", type=Path, help="Aggregate metrics JSON path.")
    parser.add_argument("--predictions-out", type=Path, help="Per-example evaluated predictions JSONL path.")
    parser.add_argument("--failure-breakdown-out", type=Path, help="Failure-count JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_table(args.predictions)
    dataset = load_table(args.dataset) if args.dataset else None
    records = merge_prediction_and_dataset(predictions, dataset)
    metrics, evaluated = evaluate_records(records)

    if args.metrics_out:
        write_json(args.metrics_out, metrics)
    if args.predictions_out:
        write_jsonl(args.predictions_out, evaluated)
    if args.failure_breakdown_out:
        write_json(args.failure_breakdown_out, metrics["failure_counts"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Expand full-path examples into state-to-next-action supervision.

Pipeline stage: core dataset construction for simulator-controlled stepwise decoding.
Inputs: full path-planning records with worlds, starts, goals, and shortest paths.
Outputs: JSONL rows where each state asks for the next action toward the goal.
Typical caller: scripts/run_phase2_stepwise_policy.sh and DAgger dataset builders.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json, write_jsonl


DEFAULT_TEST_SPLITS = ["test_size_8x8", "test_size_9x9", "test_size_12x12", "test_size_14x14", "test_size_16x16"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ACTION_ORDER = ["up", "down", "left", "right"]


def obstacle_text(sample: dict[str, Any]) -> str:
    obstacles = sample.get("obstacles", [])
    if not obstacles:
        return "none"
    return ", ".join(f"({row},{col})" for row, col in obstacles)


def is_valid_cell(sample: dict[str, Any], cell: tuple[int, int]) -> bool:
    rows, cols = sample["grid_size"]
    obstacles = {tuple(item) for item in sample.get("obstacles", [])}
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols and cell not in obstacles


def move(current: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    return current[0] + dr, current[1] + dc


def shortest_distances(sample: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Compute shortest distance from every reachable cell to the goal.

    These distances are used as optional progress features in stepwise-policy
    prompts and targets.
    """
    goal = tuple(sample["goal"])
    queue = deque([(goal, 0)])
    distances = {goal: 0}
    while queue:
        current, distance = queue.popleft()
        for action in ACTION_ORDER:
            nxt = move(current, action)
            if not is_valid_cell(sample, nxt) or nxt in distances:
                continue
            distances[nxt] = distance + 1
            queue.append((nxt, distance + 1))
    return distances


def distance_features(
    sample: dict[str, Any],
    current: tuple[int, int],
    distances: dict[tuple[int, int], int] | None = None,
) -> dict[str, int | None]:
    goal = tuple(sample["goal"])
    if distances is None:
        distances = shortest_distances(sample)
    return {
        "manhattan_distance_to_goal": abs(current[0] - goal[0]) + abs(current[1] - goal[1]),
        "shortest_remaining_steps": distances.get(current),
    }


def legal_actions(sample: dict[str, Any], current: tuple[int, int]) -> list[str]:
    return [action for action in ACTION_ORDER if is_valid_cell(sample, move(current, action))]


def legal_action_text(sample: dict[str, Any], current: tuple[int, int]) -> str:
    actions = legal_actions(sample, current)
    return "none" if not actions else ", ".join(actions)


def history_text(history: list[tuple[int, int]] | list[list[int]] | None, history_window: int) -> str:
    if not history:
        return "none"
    recent = history[-max(1, history_window) :]
    return " -> ".join(f"({int(row)},{int(col)})" for row, col in recent)


def state_question(
    sample: dict[str, Any],
    current: list[int],
    include_distance_signals: bool = False,
    include_valid_actions: bool = False,
    include_history: bool = False,
    history: list[tuple[int, int]] | list[list[int]] | None = None,
    history_window: int = 4,
    distances: dict[tuple[int, int], int] | None = None,
) -> str:
    rows, cols = sample["grid_size"]
    goal = sample["goal"]
    question = (
        f"Grid: {rows} by {cols}. Current at ({current[0]},{current[1]}). "
        f"Goal at ({goal[0]},{goal[1]}). Obstacles: {obstacle_text(sample)}. "
    )
    if include_distance_signals:
        features = distance_features(sample, tuple(current), distances=distances)
        remaining = features["shortest_remaining_steps"]
        remaining_text = "unreachable" if remaining is None else str(remaining)
        question += (
            f"Manhattan distance to goal: {features['manhattan_distance_to_goal']}. "
            f"Shortest remaining steps: {remaining_text}. "
        )
    if include_valid_actions:
        question += f"Legal actions from current state: {legal_action_text(sample, tuple(current))}. "
    if include_history:
        question += f"Recent positions: {history_text(history, history_window)}. "
    question += "Return exactly one next action from up, down, left, right, or stop."
    return question


def target_text(
    action: str,
    current: tuple[int, int],
    distances: dict[tuple[int, int], int] | None,
    target_format: str,
) -> str:
    if target_format == "action":
        return action
    next_cell = current if action == "stop" else move(current, action)
    next_distance = None if distances is None else distances.get(next_cell)
    distance_text = "unreachable" if next_distance is None else str(next_distance)
    if target_format == "action_next_distance":
        return f"{action} next ({next_cell[0]},{next_cell[1]}) distance {distance_text}"
    raise ValueError(f"unknown target format: {target_format}")


def prompt_variant(include_distance_signals: bool, include_valid_actions: bool, include_history: bool = False) -> str:
    parts = []
    if include_distance_signals:
        parts.append("distance_signals")
    if include_valid_actions:
        parts.append("valid_actions")
    if include_history:
        parts.append("history")
    return "+".join(parts) if parts else "base"


def expand_record(
    sample: dict[str, Any],
    split: str,
    include_distance_signals: bool = False,
    include_valid_actions: bool = False,
    include_history: bool = False,
    history_window: int = 4,
    target_format: str = "action",
) -> list[dict[str, Any]]:
    """Convert one full-path record into state-to-next-action examples.

    Each output row represents the state before one gold action. This is the
    dataset bridge from direct sequence generation to executor-controlled
    stepwise decoding.
    """
    coords = sample.get("solution_coordinates")
    actions = str(sample.get("agent_as_a_point", sample.get("target", ""))).split()
    if not isinstance(coords, list) or len(coords) != len(actions) + 1:
        raise ValueError(f"{sample.get('id', sample.get('example_id'))}: invalid path/action trace")

    rows = []
    base_id = sample.get("id", sample.get("example_id"))
    distances = shortest_distances(sample) if include_distance_signals or target_format != "action" else None
    for step, action in enumerate(actions):
        current = coords[step]
        row = dict(sample)
        row["id"] = f"{base_id}_step{step:03d}"
        row["example_id"] = row["id"]
        row["split"] = split
        row["step_index"] = step
        row["current"] = current
        row["question"] = state_question(
            sample,
            current,
            include_distance_signals=include_distance_signals,
            include_valid_actions=include_valid_actions,
            include_history=include_history,
            history=coords[: step + 1],
            history_window=history_window,
            distances=distances,
        )
        row["nl_description"] = row["question"]
        row["target"] = target_text(action, tuple(current), distances, target_format)
        row["stepwise_prompt_variant"] = prompt_variant(include_distance_signals, include_valid_actions, include_history)
        row["stepwise_target_format"] = target_format
        rows.append(row)

    final = dict(sample)
    final["id"] = f"{base_id}_step{len(actions):03d}"
    final["example_id"] = final["id"]
    final["split"] = split
    final["step_index"] = len(actions)
    final["current"] = coords[-1]
    final["question"] = state_question(
        sample,
        coords[-1],
        include_distance_signals=include_distance_signals,
        include_valid_actions=include_valid_actions,
        include_history=include_history,
        history=coords,
        history_window=history_window,
        distances=distances,
    )
    final["nl_description"] = final["question"]
    final["target"] = target_text("stop", tuple(coords[-1]), distances, target_format)
    final["stepwise_prompt_variant"] = prompt_variant(include_distance_signals, include_valid_actions, include_history)
    final["stepwise_target_format"] = target_format
    rows.append(final)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for row in rows:
        action_counts[row["target"]] = action_counts.get(row["target"], 0) + 1
        by_shape[row["grid_shape"]] = by_shape.get(row["grid_shape"], 0) + 1
    return {
        "examples": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "by_shape": dict(sorted(by_shape.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stepwise policy dataset.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/generated/phase2_size_generalization/react_trace_compact_sizegen/jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/phase2_size_generalization/stepwise_policy"),
    )
    parser.add_argument("--test-splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--include-distance-signals", action="store_true")
    parser.add_argument("--include-valid-actions", action="store_true")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--target-format", choices=["action", "action_next_distance"], default="action")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_names = ["train", "val", *args.test_splits]
    manifest: dict[str, Any] = {
        "source_dir": str(args.source_dir),
        "splits": {},
        "test_splits": args.test_splits,
        "include_distance_signals": args.include_distance_signals,
        "include_valid_actions": args.include_valid_actions,
        "include_history": args.include_history,
        "history_window": args.history_window,
        "target_format": args.target_format,
    }
    for split in split_names:
        original_rows = load_records(args.source_dir / f"{split}.jsonl")
        expanded = []
        for row in original_rows:
            expanded.extend(
                expand_record(
                row,
                split,
                include_distance_signals=args.include_distance_signals,
                include_valid_actions=args.include_valid_actions,
                include_history=args.include_history,
                history_window=args.history_window,
                target_format=args.target_format,
            )
            )
        write_jsonl(args.output_dir / "jsonl" / f"{split}.jsonl", expanded)
        manifest["splits"][split] = {
            "source_examples": len(original_rows),
            "stepwise_examples": len(expanded),
            **summarize(expanded),
        }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

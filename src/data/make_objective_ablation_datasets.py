#!/usr/bin/env python3
"""Create SFT objective-ablation datasets from the same base examples.

Pipeline stage: dataset conversion for comparing supervision objectives.
Inputs: Phase 0 records and objective-format options.
Outputs: JSONL datasets for single-reference, multi-reference, and auxiliary-mixture training.
Typical caller: scripts/run_phase2_objective_ablation.sh.
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

from src.data.ppnl_io import load_records, normalize_action_text, write_json, write_jsonl
from src.models.run_phase1_baselines import DEFAULT_TEST_SPLITS


ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ACTION_ORDER = ["up", "down", "left", "right"]
METHODS = ("single_reference", "multi_reference", "auxiliary_mixture")


def base_question(record: dict[str, Any]) -> str:
    return str(record.get("question", record.get("nl_description", record.get("natural_input", "")))).strip()


def move(position: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    return position[0] + dr, position[1] + dc


def is_valid_cell(record: dict[str, Any], cell: tuple[int, int]) -> bool:
    rows, cols = record["grid_size"]
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols and list(cell) not in record.get("obstacles", [])


def shortest_distances(record: dict[str, Any]) -> dict[tuple[int, int], int]:
    goal = tuple(record["goal"])
    queue = deque([(goal, 0)])
    distances = {goal: 0}
    while queue:
        current, distance = queue.popleft()
        for action in ACTION_ORDER:
            nxt = move(current, action)
            if not is_valid_cell(record, nxt) or nxt in distances:
                continue
            distances[nxt] = distance + 1
            queue.append((nxt, distance + 1))
    return distances


def enumerate_shortest_paths(record: dict[str, Any], max_refs: int) -> list[list[str]]:
    start = tuple(record["start"])
    goal = tuple(record["goal"])
    distances = shortest_distances(record)
    if start not in distances:
        return [normalize_action_text(record["target"]).split()]
    paths: list[list[str]] = []

    def dfs(current: tuple[int, int], actions: list[str]) -> None:
        if len(paths) >= max_refs:
            return
        if current == goal:
            paths.append(list(actions))
            return
        current_distance = distances[current]
        for action in ACTION_ORDER:
            nxt = move(current, action)
            if distances.get(nxt) == current_distance - 1:
                actions.append(action)
                dfs(nxt, actions)
                actions.pop()

    dfs(start, [])
    return paths or [normalize_action_text(record["target"]).split()]


def with_question_target(record: dict[str, Any], question: str, target: str, task: str, suffix: str) -> dict[str, Any]:
    row = dict(record)
    base_id = row.get("example_id", row.get("id"))
    row["base_example_id"] = base_id
    row["id"] = f"{base_id}_{suffix}"
    row["example_id"] = f"{base_id}_{suffix}"
    row["objective_task"] = task
    row["question"] = question
    row["nl_description"] = question
    row["target"] = target
    return row


def invalid_action(record: dict[str, Any]) -> str:
    start = tuple(record["start"])
    for action in ACTION_ORDER:
        if not is_valid_cell(record, move(start, action)):
            return action
    oracle_first = normalize_action_text(record["target"]).split()[0]
    for action in ACTION_ORDER:
        if action != oracle_first:
            return action
    return "up"


def partial_path_task(record: dict[str, Any]) -> tuple[str, str] | None:
    actions = normalize_action_text(record["target"]).split()
    coords = record.get("solution_coordinates")
    if len(actions) < 2 or not isinstance(coords, list):
        return None
    prefix_len = max(1, len(actions) // 2)
    prefix = " ".join(actions[:prefix_len])
    current = coords[prefix_len]
    question = (
        f"{base_question(record)} Partial path already taken: {prefix}. "
        f"Current position: ({current[0]},{current[1]}). Return only the next action."
    )
    return question, actions[prefix_len]


def corrupted_path(record: dict[str, Any]) -> str:
    actions = normalize_action_text(record["target"]).split()
    bad = invalid_action(record)
    if actions:
        return " ".join([bad, *actions[1:]])
    return bad


def single_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(record, objective_task="full_path", base_example_id=record.get("example_id", record.get("id")))
        for record in records
    ]


def multi_reference_records(records: list[dict[str, Any]], max_refs: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for ref_index, actions in enumerate(enumerate_shortest_paths(record, max_refs)):
            rows.append(
                with_question_target(
                    record,
                    base_question(record),
                    " ".join(actions),
                    "full_path_multi_reference",
                    f"ref{ref_index}",
                )
            )
    return rows


def auxiliary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        question = base_question(record)
        actions = normalize_action_text(record["target"]).split()
        if not actions:
            continue
        rows.append(with_question_target(record, question, " ".join(actions), "full_path", "full"))
        rows.append(
            with_question_target(
                record,
                f"{question} Return only the next action from the start.",
                actions[0],
                "next_action_from_start",
                "next0",
            )
        )
        rows.append(
            with_question_target(
                record,
                f"{question} Return only the shortest path length as an integer.",
                str(record["shortest_path_length"]),
                "shortest_path_length",
                "length",
            )
        )
        valid_action = actions[0]
        rows.append(
            with_question_target(
                record,
                f"{question} Candidate first action: {valid_action}. Is this action valid? Answer valid or invalid.",
                "valid",
                "candidate_action_validity",
                "valid_action",
            )
        )
        bad_action = invalid_action(record)
        rows.append(
            with_question_target(
                record,
                f"{question} Candidate first action: {bad_action}. Is this action valid? Answer valid or invalid.",
                "invalid",
                "candidate_action_validity",
                "invalid_action",
            )
        )
        partial = partial_path_task(record)
        if partial is not None:
            partial_question, next_action = partial
            rows.append(with_question_target(record, partial_question, next_action, "partial_path_next_action", "partial_next"))
        oracle_path = " ".join(actions)
        rows.append(
            with_question_target(
                record,
                f"{question} Generated path: {oracle_path}. Is this path feasible? Answer feasible or infeasible.",
                "feasible",
                "generated_path_feasibility",
                "path_feasible",
            )
        )
        bad_path = corrupted_path(record)
        rows.append(
            with_question_target(
                record,
                f"{question} Generated path: {bad_path}. Is this path feasible? Answer feasible or infeasible.",
                "infeasible",
                "generated_path_feasibility",
                "path_infeasible",
            )
        )
    return rows


def convert_train(records: list[dict[str, Any]], method: str, max_refs: int) -> list[dict[str, Any]]:
    if method == "single_reference":
        return single_reference_records(records)
    if method == "multi_reference":
        return multi_reference_records(records, max_refs)
    if method == "auxiliary_mixture":
        return auxiliary_records(records)
    raise ValueError(method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create objective-ablation datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/generated/phase2_objective_ablation"))
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-refs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_records(args.source_dir / "train.jsonl")
    val = load_records(args.source_dir / "val.jsonl")
    manifest: dict[str, Any] = {"source_dir": str(args.source_dir), "methods": {}}
    for method in args.methods:
        root = args.output_root / method / "jsonl"
        train_rows = convert_train(train, method, args.max_refs)
        write_jsonl(root / "train.jsonl", train_rows)
        write_jsonl(root / "val.jsonl", single_reference_records(val))
        for split in DEFAULT_TEST_SPLITS:
            write_jsonl(root / f"{split}.jsonl", single_reference_records(load_records(args.source_dir / f"{split}.jsonl")))
        unique_pairs = {row.get("base_example_id", row.get("id")) for row in train_rows}
        task_counts: dict[str, int] = {}
        for row in train_rows:
            task_counts[row["objective_task"]] = task_counts.get(row["objective_task"], 0) + 1
        manifest["methods"][method] = {
            "root": str(args.output_root / method),
            "train_rows": len(train_rows),
            "base_train_pairs": len(train),
            "unique_train_pairs": len(unique_pairs),
            "task_counts": task_counts,
        }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

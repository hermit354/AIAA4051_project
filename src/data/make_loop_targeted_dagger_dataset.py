#!/usr/bin/env python3
"""Build loop-targeted DAgger data from residual rollout failures.

Pipeline stage: targeted data construction after failure analysis identifies looping behavior.
Inputs: evaluated rollout traces and optional history/distance settings.
Outputs: history-aware state-to-next-action JSONL train and validation splits.
Typical caller: scripts/run_phase2_loop_dagger_history.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make_stepwise_policy_dataset import (
    ACTION_ORDER,
    expand_record,
    is_valid_cell,
    move,
    shortest_distances,
    state_question,
    target_text,
)
from src.data.ppnl_io import load_records, write_json, write_jsonl


def ensure_grid_size(row: dict[str, Any]) -> dict[str, Any]:
    if "grid_size" not in row:
        world = row.get("world")
        if isinstance(world, list) and world:
            row["grid_size"] = [len(world), len(world[0])]
        else:
            difficulty = row.get("difficulty", {})
            if "grid_rows" in difficulty and "grid_cols" in difficulty:
                row["grid_size"] = [int(difficulty["grid_rows"]), int(difficulty["grid_cols"])]
    return row


def oracle_action(
    sample: dict[str, Any],
    current: tuple[int, int],
    previous: tuple[int, int] | None,
    distances: dict[tuple[int, int], int],
) -> str | None:
    if current == tuple(sample["goal"]):
        return "stop"
    current_distance = distances.get(current)
    if current_distance is None:
        return None
    decreasing = []
    for action in ACTION_ORDER:
        nxt = move(current, action)
        if not is_valid_cell(sample, nxt):
            continue
        if distances.get(nxt) == current_distance - 1:
            decreasing.append((action, nxt))
    for action, nxt in decreasing:
        if previous is None or nxt != previous:
            return action
    if decreasing:
        return decreasing[0][0]
    non_backtracking = [
        (action, move(current, action))
        for action in ACTION_ORDER
        if is_valid_cell(sample, move(current, action)) and (previous is None or move(current, action) != previous)
    ]
    if not non_backtracking:
        return None
    return min(non_backtracking, key=lambda item: distances.get(item[1], 1_000_000))[0]


def is_loop_step(trajectory: list[tuple[int, int]], step: int, recent_window: int) -> bool:
    if step <= 0 or step >= len(trajectory):
        return False
    current = trajectory[step]
    nxt = trajectory[step + 1] if step + 1 < len(trajectory) else None
    if nxt is None:
        return False
    if step >= 1 and nxt == trajectory[step - 1]:
        return True
    recent = set(trajectory[max(0, step - recent_window + 1) : step + 1])
    return nxt in recent


def correction_record(
    row: dict[str, Any],
    current: tuple[int, int],
    history: list[tuple[int, int]],
    target: str,
    correction_index: int,
    target_format: str,
    include_history: bool,
    history_window: int,
    distances: dict[tuple[int, int], int],
) -> dict[str, Any]:
    sample = ensure_grid_size(dict(row))
    base_id = sample.get("id", sample.get("example_id"))
    sample["id"] = f"{base_id}_loopdagger{correction_index:04d}"
    sample["example_id"] = sample["id"]
    sample["base_example_id"] = base_id
    sample["current"] = list(current)
    sample["question"] = state_question(
        sample,
        list(current),
        include_distance_signals=True,
        include_history=include_history,
        history=history if include_history else None,
        history_window=history_window,
        distances=distances,
    )
    sample["nl_description"] = sample["question"]
    sample["target"] = target_text(target, current, distances, target_format)
    sample["objective_task"] = "loop_targeted_dagger"
    sample["stepwise_prompt_variant"] = "distance_signals+history" if include_history else "distance_signals"
    sample["stepwise_target_format"] = target_format
    return sample


def collect_loop_corrections(
    evaluated_paths: list[Path],
    target_format: str,
    include_history: bool,
    history_window: int,
    recent_window: int,
    max_corrections_per_example: int,
) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int], str]] = set()
    for path in evaluated_paths:
        for row in load_records(path):
            if row.get("success"):
                continue
            row = ensure_grid_size(row)
            trajectory = [tuple(item) for item in row.get("executed_trajectory", [])]
            if len(trajectory) < 2:
                continue
            distances = shortest_distances(row)
            added = 0
            for step in range(len(trajectory) - 1):
                if not is_loop_step(trajectory, step, recent_window):
                    continue
                current = trajectory[step]
                previous = trajectory[step - 1] if step > 0 else None
                target = oracle_action(row, current, previous, distances)
                if target is None:
                    continue
                key = (str(row.get("id")), current, target)
                if key in seen:
                    continue
                corrections.append(
                    correction_record(
                        row,
                        current,
                        trajectory[: step + 1],
                        target,
                        len(corrections),
                        target_format,
                        include_history,
                        history_window,
                        distances,
                    )
                )
                seen.add(key)
                added += 1
                if added >= max_corrections_per_example:
                    break
    return corrections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create loop-targeted DAgger stepwise data.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--evaluated", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-format", choices=["action", "action_next_distance"], default="action_next_distance")
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=4)
    parser.add_argument("--max-corrections-per-example", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_source = load_records(args.source_dir / "train.jsonl")
    val_source = load_records(args.source_dir / "val.jsonl")

    oracle_train = []
    for row in train_source:
        oracle_train.extend(
            expand_record(
                row,
                "train",
                include_distance_signals=True,
                include_history=args.include_history,
                history_window=args.history_window,
                target_format=args.target_format,
            )
        )
    corrections = collect_loop_corrections(
        args.evaluated,
        target_format=args.target_format,
        include_history=args.include_history,
        history_window=args.history_window,
        recent_window=args.recent_window,
        max_corrections_per_example=args.max_corrections_per_example,
    )
    val_rows = []
    for row in val_source:
        val_rows.extend(
            expand_record(
                row,
                "val",
                include_distance_signals=True,
                include_history=args.include_history,
                history_window=args.history_window,
                target_format=args.target_format,
            )
        )

    train_rows = [*oracle_train, *corrections]
    write_jsonl(args.output_dir / "jsonl" / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "jsonl" / "val.jsonl", val_rows)
    manifest = {
        "source_dir": str(args.source_dir),
        "evaluated": [str(path) for path in args.evaluated],
        "target_format": args.target_format,
        "include_history": args.include_history,
        "history_window": args.history_window,
        "recent_window": args.recent_window,
        "max_corrections_per_example": args.max_corrections_per_example,
        "oracle_train_examples": len(oracle_train),
        "loop_corrections": len(corrections),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "note": "Corrections are collected from residual rollout failures; keep this experiment separate from clean IID/OOD evaluation if evaluated files are test splits.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

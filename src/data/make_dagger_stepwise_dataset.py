#!/usr/bin/env python3
"""Create DAgger-style stepwise correction data from rollout trajectories.

Pipeline stage: data construction after an initial stepwise model has been evaluated.
Inputs: evaluated rollout JSONL files with trajectories and failure context.
Outputs: state-to-next-action JSONL train and validation examples.
Typical caller: scripts/run_phase2_stepwise_dagger.sh or related DAgger launchers.
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
    ACTION_DELTAS,
    ACTION_ORDER,
    expand_record,
    is_valid_cell,
    move,
    shortest_distances,
    state_question,
)
from src.data.ppnl_io import load_records, normalize_action_text, write_json, write_jsonl


def oracle_action(sample: dict[str, Any], current: tuple[int, int]) -> str | None:
    goal = tuple(sample["goal"])
    if current == goal:
        return "stop"
    distances = shortest_distances(sample)
    current_distance = distances.get(current)
    if current_distance is None:
        return None
    for action in ACTION_ORDER:
        nxt = move(current, action)
        if distances.get(nxt) == current_distance - 1:
            return action
    return None


def action_is_valid(sample: dict[str, Any], current: tuple[int, int], action: str) -> bool:
    return action in ACTION_DELTAS and is_valid_cell(sample, move(current, action))


def correction_record(
    sample: dict[str, Any],
    current: tuple[int, int],
    target: str,
    step: int,
    correction_index: int,
    include_distance_signals: bool,
    distances: dict[tuple[int, int], int] | None,
) -> dict[str, Any]:
    row = dict(sample)
    base_id = row.get("id", row.get("example_id"))
    row["id"] = f"{base_id}_dagger{correction_index:03d}"
    row["example_id"] = row["id"]
    row["base_example_id"] = base_id
    row["step_index"] = step
    row["current"] = list(current)
    row["target"] = target
    row["question"] = state_question(
        row,
        list(current),
        include_distance_signals=include_distance_signals,
        distances=distances,
    )
    row["nl_description"] = row["question"]
    row["objective_task"] = "dagger_correction"
    row["stepwise_prompt_variant"] = "distance_signals" if include_distance_signals else "base"
    return row


def collect_corrections(
    source_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    include_distance_signals: bool,
    max_corrections_per_example: int,
) -> list[dict[str, Any]]:
    by_id = {str(row.get("id", row.get("example_id", index))): row for index, row in enumerate(source_rows)}
    corrections: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int], str]] = set()

    for index, rollout in enumerate(rollout_rows):
        sample = by_id.get(str(rollout.get("id")))
        if sample is None:
            sample = source_rows[index]
        current = tuple(sample["start"])
        distances = shortest_distances(sample) if include_distance_signals else None
        added = 0
        actions = normalize_action_text(rollout.get("generated", rollout.get("model_output", ""))).split()
        for step, action in enumerate(actions):
            oracle = oracle_action(sample, current)
            if oracle is None:
                break
            if action != oracle:
                key = (str(sample.get("id", sample.get("example_id"))), current, oracle)
                if key not in seen:
                    corrections.append(
                        correction_record(
                            sample,
                            current,
                            oracle,
                            step,
                            len(corrections),
                            include_distance_signals,
                            distances,
                        )
                    )
                    seen.add(key)
                    added += 1
                if added >= max_corrections_per_example:
                    break
            if action == "stop" or current == tuple(sample["goal"]) or not action_is_valid(sample, current, action):
                break
            current = move(current, action)
    return corrections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DAgger-style stepwise correction data.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--rollout-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-distance-signals", action="store_true")
    parser.add_argument("--max-corrections-per-example", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_source = load_records(args.source_dir / "train.jsonl")
    val_source = load_records(args.source_dir / "val.jsonl")
    rollout_rows = load_records(args.rollout_predictions)

    oracle_train = []
    for row in train_source:
        oracle_train.extend(expand_record(row, "train", include_distance_signals=args.include_distance_signals))
    corrections = collect_corrections(
        train_source,
        rollout_rows,
        include_distance_signals=args.include_distance_signals,
        max_corrections_per_example=args.max_corrections_per_example,
    )
    train_rows = [*oracle_train, *corrections]

    val_rows = []
    for row in val_source:
        val_rows.extend(expand_record(row, "val", include_distance_signals=args.include_distance_signals))

    write_jsonl(args.output_dir / "jsonl" / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "jsonl" / "val.jsonl", val_rows)
    manifest: dict[str, Any] = {
        "source_dir": str(args.source_dir),
        "rollout_predictions": str(args.rollout_predictions),
        "include_distance_signals": args.include_distance_signals,
        "oracle_train_examples": len(oracle_train),
        "dagger_corrections": len(corrections),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "max_corrections_per_example": args.max_corrections_per_example,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

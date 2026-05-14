#!/usr/bin/env python3
"""Validate Phase 0 generated datasets against project invariants.

Pipeline stage: data-quality gate before training and evaluation.
Inputs: generated Phase 0 JSON/JSONL splits and expected split definitions.
Outputs: validation JSON reports and terminal summaries.
Typical caller: baseline launch scripts before model training starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, write_json


EXPECTED = {
    "train": {"examples": 16032, "envs": 668, "env_obs": {1: 28, 2: 160, 3: 160, 4: 160, 5: 160}},
    "val": {"examples": 2004, "envs": 668, "env_obs": {1: 28, 2: 160, 3: 160, 4: 160, 5: 160}},
    "test_unseen_placement": {
        "examples": 2004,
        "envs": 668,
        "env_obs": {1: 28, 2: 160, 3: 160, 4: 160, 5: 160},
    },
    "test_unseen_environment": {
        "examples": 5040,
        "envs": 168,
        "env_obs": {1: 8, 2: 40, 3: 40, 4: 40, 5: 40},
    },
    **{
        f"ood_size_{size}x{size}": {
            "examples": 3750,
            "envs": 125,
            "env_obs": {1: 25, 2: 25, 3: 25, 4: 25, 5: 25},
        }
        for size in (5, 7, 8, 9, 10)
    },
    "ood_dense_6x6": {"examples": 4500, "envs": 150, "env_obs": {6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25}},
    "ood_dense_5x5": {"examples": 4500, "envs": 150, "env_obs": {6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25}},
    "ood_dense_7x7": {"examples": 4500, "envs": 150, "env_obs": {6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25}},
    "ood_aspect_10x5": {"examples": 3750, "envs": 125, "env_obs": {1: 25, 2: 25, 3: 25, 4: 25, 5: 25}},
    "ood_aspect_6x9": {"examples": 3750, "envs": 125, "env_obs": {1: 25, 2: 25, 3: 25, 4: 25, 5: 25}},
}

ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def shortest_path_length(
    rows: int,
    cols: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> int | None:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current, distance = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if nxt in obstacles or nxt in visited:
                continue
            if nxt == goal:
                return distance + 1
            visited.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def add_failure(failures: list[str], message: str) -> None:
    failures.append(message)


def pair_key(row: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (tuple(row["start"]), tuple(row["goal"]))


def env_distribution(rows: list[dict[str, Any]]) -> dict[int, int]:
    env_obs: dict[str, int] = {}
    for row in rows:
        env_obs[row["environment_id"]] = int(row["obstacle_count"])
    counts: dict[int, int] = {}
    for obstacle_count in env_obs.values():
        counts[obstacle_count] = counts.get(obstacle_count, 0) + 1
    return dict(sorted(counts.items()))


def validate_example(row: dict[str, Any], failures: list[str]) -> None:
    required = [
        "example_id",
        "split",
        "grid_size",
        "obstacle_count",
        "environment_id",
        "start",
        "goal",
        "obstacles",
        "shortest_path_length",
        "manhattan_distance",
        "natural_input",
        "structured_input",
        "grid_matrix_input",
        "world",
        "target",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        add_failure(failures, f"{row.get('example_id', '<unknown>')}: missing keys {missing}")
        return

    rows, cols = row["grid_size"]
    obstacles = {tuple(item) for item in row["obstacles"]}
    if len(obstacles) != int(row["obstacle_count"]):
        add_failure(failures, f"{row['example_id']}: obstacle_count mismatch")
    for point_name in ("start", "goal"):
        point = tuple(row[point_name])
        if point in obstacles:
            add_failure(failures, f"{row['example_id']}: {point_name} overlaps obstacle")
        if not (0 <= point[0] < rows and 0 <= point[1] < cols):
            add_failure(failures, f"{row['example_id']}: {point_name} out of bounds")
    if row["start"] == row["goal"]:
        add_failure(failures, f"{row['example_id']}: start equals goal")

    bfs_shortest = shortest_path_length(rows, cols, obstacles, tuple(row["start"]), tuple(row["goal"]))
    if bfs_shortest is None:
        add_failure(failures, f"{row['example_id']}: no reachable path by independent BFS")
    elif bfs_shortest != int(row["shortest_path_length"]):
        add_failure(
            failures,
            f"{row['example_id']}: shortest_path_length should be {bfs_shortest}, found {row['shortest_path_length']}",
        )

    current = tuple(row["start"])
    trajectory = [current]
    for action in str(row["target"]).split():
        if action not in ACTION_DELTAS:
            add_failure(failures, f"{row['example_id']}: invalid target token {action}")
            return
        dr, dc = ACTION_DELTAS[action]
        current = (current[0] + dr, current[1] + dc)
        trajectory.append(current)
        if not (0 <= current[0] < rows and 0 <= current[1] < cols):
            add_failure(failures, f"{row['example_id']}: target path goes out of bounds")
            return
        if current in obstacles:
            add_failure(failures, f"{row['example_id']}: target path hits obstacle")
            return
    if list(current) != row["goal"]:
        add_failure(failures, f"{row['example_id']}: target path does not reach goal")
    if len(trajectory) - 1 != int(row["shortest_path_length"]):
        add_failure(failures, f"{row['example_id']}: target path length mismatch")


def validate_split(name: str, rows: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    expected = EXPECTED[name]
    env_ids = {row["environment_id"] for row in rows}
    if len(rows) != expected["examples"]:
        add_failure(failures, f"{name}: expected {expected['examples']} examples, found {len(rows)}")
    if len(env_ids) != expected["envs"]:
        add_failure(failures, f"{name}: expected {expected['envs']} envs, found {len(env_ids)}")
    distribution = env_distribution(rows)
    if distribution != expected["env_obs"]:
        add_failure(failures, f"{name}: expected env obstacle distribution {expected['env_obs']}, found {distribution}")
    for row in rows:
        if row.get("split") != name:
            add_failure(failures, f"{row.get('example_id')}: split field should be {name}")
        validate_example(row, failures)
    return {"examples": len(rows), "environments": len(env_ids), "env_obstacle_distribution": distribution}


def validate_pair_non_overlap(splits: dict[str, list[dict[str, Any]]], failures: list[str]) -> None:
    seen_pairs: dict[str, dict[str, set[tuple[tuple[int, int], tuple[int, int]]]]] = defaultdict(lambda: defaultdict(set))
    for split in ("train", "val", "test_unseen_placement"):
        for row in splits[split]:
            seen_pairs[row["environment_id"]][split].add(pair_key(row))

    for env_id, by_split in seen_pairs.items():
        for left, right in (("train", "val"), ("train", "test_unseen_placement"), ("val", "test_unseen_placement")):
            overlap = by_split[left] & by_split[right]
            if overlap:
                add_failure(failures, f"{env_id}: pair overlap between {left} and {right}: {sorted(overlap)[:3]}")


def load_splits(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    split_dir = dataset_dir / "jsonl"
    if not split_dir.exists():
        split_dir = dataset_dir
    splits = {}
    for split in EXPECTED:
        path = split_dir / f"{split}.jsonl"
        if not path.exists():
            alt = dataset_dir / "json" / f"{split}.json"
            if not alt.exists():
                raise FileNotFoundError(f"missing split file for {split}: {path}")
            path = alt
        splits[split] = load_records(path)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated Phase 0 data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/generated/phase0"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/phase0_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_splits(args.dataset_dir)
    failures: list[str] = []
    report = {"splits": {}}
    for split, rows in splits.items():
        report["splits"][split] = validate_split(split, rows, failures)
    validate_pair_non_overlap(splits, failures)
    report["passed"] = not failures
    report["failures"] = failures
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

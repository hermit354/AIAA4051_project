#!/usr/bin/env python3
"""Generate the base Phase 0 grid path-planning datasets.

Pipeline stage: first data-generation step for the whole project.
Inputs: split specifications, obstacle-count distributions, and a random seed.
Outputs: JSON/JSONL train, validation, ID test, and OOD test splits with shortest-path targets.
Typical caller: a direct CLI command or baseline experiment launchers before training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import write_json, write_jsonl


ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for one generated split.

    The generator uses this object to keep split names, grid shapes, obstacle
    distributions, and pair counts together. That makes the Phase 0 data design
    explicit and easy to discuss during project review.
    """

    name: str
    split: str
    rows: int
    cols: int
    obstacle_env_counts: dict[int, int]
    pairs_per_env: int
    dense: bool = False


ID_TRAIN_COUNTS = {1: 28, 2: 160, 3: 160, 4: 160, 5: 160}
ID_UNSEEN_COUNTS = {1: 8, 2: 40, 3: 40, 4: 40, 5: 40}
STANDARD_OOD_COUNTS = {1: 25, 2: 25, 3: 25, 4: 25, 5: 25}
DENSE_OOD_COUNTS = {6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25}

OOD_SPECS = [
    DatasetSpec(f"ood_size_{size}x{size}", f"ood_size_{size}x{size}", size, size, STANDARD_OOD_COUNTS, 30)
    for size in (5, 7, 8, 9, 10)
] + [
    DatasetSpec("ood_dense_6x6", "ood_dense_6x6", 6, 6, DENSE_OOD_COUNTS, 30, dense=True),
    DatasetSpec("ood_dense_5x5", "ood_dense_5x5", 5, 5, DENSE_OOD_COUNTS, 30, dense=True),
    DatasetSpec("ood_dense_7x7", "ood_dense_7x7", 7, 7, DENSE_OOD_COUNTS, 30, dense=True),
    DatasetSpec("ood_aspect_10x5", "ood_aspect_10x5", 10, 5, STANDARD_OOD_COUNTS, 30),
    DatasetSpec("ood_aspect_6x9", "ood_aspect_6x9", 6, 9, STANDARD_OOD_COUNTS, 30),
]


def shape_tag(rows: int, cols: int) -> str:
    return f"{rows}x{cols}"


def environment_id(rows: int, cols: int, obstacle_count: int, index: int, prefix: str = "env") -> str:
    return f"{shape_tag(rows, cols)}_obs{obstacle_count}_{prefix}{index:03d}"


def obstacle_key(obstacles: list[list[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(row), int(col)) for row, col in obstacles))


def free_cells(rows: int, cols: int, obstacles: list[list[int]]) -> list[tuple[int, int]]:
    blocked = {tuple(item) for item in obstacles}
    return [(row, col) for row in range(rows) for col in range(cols) if (row, col) not in blocked]


def shortest_path(
    rows: int,
    cols: int,
    obstacles: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Return an optimal path on an unweighted grid, or None if unreachable.

    BFS is used because every move has unit cost. The returned path becomes the
    supervised target and the reference for later optimality checks.
    """
    blocked = {tuple(item) for item in obstacles}
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        for dr, dc in ACTION_DELTAS.values():
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def path_to_actions(path: list[tuple[int, int]]) -> list[str]:
    actions = []
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        for action, action_delta in ACTION_DELTAS.items():
            if action_delta == delta:
                actions.append(action)
                break
        else:
            raise ValueError(f"non-adjacent path transition: {current} -> {nxt}")
    return actions


def grid_matrix(rows: int, cols: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> str:
    blocked = {tuple(item) for item in obstacles}
    lines = []
    for row in range(rows):
        cells = []
        for col in range(cols):
            if [row, col] == start:
                cells.append("S")
            elif [row, col] == goal:
                cells.append("G")
            elif (row, col) in blocked:
                cells.append("X")
            else:
                cells.append(".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def natural_input(rows: int, cols: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> str:
    if obstacles:
        obstacle_text = ", ".join(f"({row},{col})" for row, col in obstacles)
    else:
        obstacle_text = "none"
    return (
        f"Grid: {rows} by {cols}. Start at ({start[0]},{start[1]}). "
        f"Goal at ({goal[0]},{goal[1]}). Obstacles: {obstacle_text}."
    )


def structured_input(rows: int, cols: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> str:
    obstacle_text = "[" + ", ".join(f"({row},{col})" for row, col in obstacles) + "]"
    return (
        f"Grid size: {rows} by {cols}. Start: ({start[0]},{start[1]}). "
        f"Goal: ({goal[0]},{goal[1]}). Obstacles: {obstacle_text}"
    )


def legacy_nl(rows: int, cols: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> str:
    if obstacles:
        obstacle_text = ", ".join(f"({row},{col})" for row, col in obstacles)
    else:
        obstacle_text = "none"
    return (
        f"You are in a {rows} by {cols} world. "
        f"There are obstacles that you have to avoid at: {obstacle_text}. "
        f"Go from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]})"
    )


def world_matrix(rows: int, cols: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> list[list[int]]:
    blocked = {tuple(item) for item in obstacles}
    world = []
    for row in range(rows):
        world_row = []
        for col in range(cols):
            world_row.append(1 if (row, col) in blocked else 0)
        world.append(world_row)
    world[start[0]][start[1]] = 2
    world[goal[0]][goal[1]] = 3
    return world


def generate_environments(
    rows: int,
    cols: int,
    obstacle_env_counts: dict[int, int],
    rng: random.Random,
    prefix: str,
) -> list[dict[str, Any]]:
    """Sample unique obstacle layouts for one grid shape.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.
        obstacle_env_counts: Mapping from obstacle count to number of layouts.
        rng: Seeded random generator used for reproducibility.
        prefix: Environment-id prefix used to distinguish split groups.

    Returns:
        Environment dictionaries without start/goal placements.
    """
    environments = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    all_cells = list(range(rows * cols))
    for obstacle_count, count in sorted(obstacle_env_counts.items()):
        if obstacle_count > rows * cols - 2:
            raise ValueError(f"{shape_tag(rows, cols)} cannot fit {obstacle_count} obstacles plus start/goal")
        generated = 0
        attempts = 0
        while generated < count and attempts < count * 2000:
            attempts += 1
            cells = rng.sample(all_cells, obstacle_count)
            obstacles = [[cell // cols, cell % cols] for cell in cells]
            key = obstacle_key(obstacles)
            scoped_key = (obstacle_count, key)
            if scoped_key in seen:
                continue
            seen.add(scoped_key)
            environments.append(
                {
                    "environment_id": environment_id(rows, cols, obstacle_count, generated, prefix),
                    "grid_size": [rows, cols],
                    "grid_shape": shape_tag(rows, cols),
                    "obstacle_count": obstacle_count,
                    "obstacles": [list(item) for item in key],
                }
            )
            generated += 1
        if generated != count:
            raise RuntimeError(f"only generated {generated}/{count} envs for {shape_tag(rows, cols)} obs={obstacle_count}")
    return environments


def generate_pairs(
    env: dict[str, Any],
    count: int,
    rng: random.Random,
    excluded: set[tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    """Sample reachable start-goal pairs for a fixed environment.

    The function rejects unreachable pairs and excluded placements so validation
    and test rows do not duplicate training placements for the same layout.
    """
    rows, cols = env["grid_size"]
    cells = free_cells(rows, cols, env["obstacles"])
    excluded = excluded or set()
    pairs = []
    seen = set(excluded)
    attempts = 0
    max_attempts = max(5000, count * 500)
    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        start, goal = rng.sample(cells, 2)
        pair_key = (start, goal)
        if pair_key in seen:
            continue
        path = shortest_path(rows, cols, env["obstacles"], start, goal)
        if path is None:
            continue
        seen.add(pair_key)
        pairs.append((start, goal, path))
    if len(pairs) != count:
        raise RuntimeError(f"{env['environment_id']} generated {len(pairs)}/{count} reachable pairs")
    return pairs


def make_example(
    env: dict[str, Any],
    split: str,
    pair_index: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
    is_dense: bool,
) -> dict[str, Any]:
    rows, cols = env["grid_size"]
    start_list = [start[0], start[1]]
    goal_list = [goal[0], goal[1]]
    actions = path_to_actions(path)
    example_id = f"{split}_{env['environment_id']}_pair{pair_index:03d}"
    return {
        "example_id": example_id,
        "id": example_id,
        "split": split,
        "grid_size": [rows, cols],
        "grid_shape": env["grid_shape"],
        "obstacle_count": env["obstacle_count"],
        "is_dense": is_dense,
        "environment_id": env["environment_id"],
        "start": start_list,
        "goal": goal_list,
        "goals": [goal_list],
        "obstacles": env["obstacles"],
        "shortest_path_length": len(actions),
        "manhattan_distance": abs(start[0] - goal[0]) + abs(start[1] - goal[1]),
        "natural_input": natural_input(rows, cols, env["obstacles"], start_list, goal_list),
        "structured_input": structured_input(rows, cols, env["obstacles"], start_list, goal_list),
        "grid_matrix_input": grid_matrix(rows, cols, env["obstacles"], start_list, goal_list),
        "nl_description": legacy_nl(rows, cols, env["obstacles"], start_list, goal_list),
        "question": legacy_nl(rows, cols, env["obstacles"], start_list, goal_list),
        "target": " ".join(actions),
        "agent_as_a_point": " ".join(actions) + (" " if actions else ""),
        "solution_coordinates": [[row, col] for row, col in path],
        "world": world_matrix(rows, cols, env["obstacles"], start_list, goal_list),
    }


def split_seen_pairs(pairs: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]) -> dict[str, Any]:
    return {
        "train": pairs[:24],
        "val": pairs[24:27],
        "test_unseen_placement": pairs[27:30],
    }


def generate_id_splits(rng: random.Random) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    seen_envs = generate_environments(6, 6, ID_TRAIN_COUNTS, rng, "seen")
    unseen_envs = generate_environments(6, 6, ID_UNSEEN_COUNTS, rng, "unseen")
    splits = {"train": [], "val": [], "test_unseen_placement": [], "test_unseen_environment": []}

    for env in seen_envs:
        pairs = generate_pairs(env, 30, rng)
        for split, split_pairs in split_seen_pairs(pairs).items():
            for pair_index, (start, goal, path) in enumerate(split_pairs):
                splits[split].append(make_example(env, split, pair_index, start, goal, path, is_dense=False))

    for env in unseen_envs:
        pairs = generate_pairs(env, 30, rng)
        for pair_index, (start, goal, path) in enumerate(pairs):
            splits["test_unseen_environment"].append(
                make_example(env, "test_unseen_environment", pair_index, start, goal, path, is_dense=False)
            )

    return splits, seen_envs + unseen_envs


def generate_ood_split(spec: DatasetSpec, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    envs = generate_environments(spec.rows, spec.cols, spec.obstacle_env_counts, rng, spec.name)
    examples = []
    for env in envs:
        for pair_index, (start, goal, path) in enumerate(generate_pairs(env, spec.pairs_per_env, rng)):
            examples.append(make_example(env, spec.split, pair_index, start, goal, path, is_dense=spec.dense))
    return examples, envs


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    obstacle_distribution: dict[str, int] = {}
    env_ids = set()
    for row in rows:
        key = str(row["obstacle_count"])
        obstacle_distribution[key] = obstacle_distribution.get(key, 0) + 1
        env_ids.add(row["environment_id"])
    return {
        "examples": len(rows),
        "environments": len(env_ids),
        "obstacle_example_distribution": dict(sorted(obstacle_distribution.items(), key=lambda item: int(item[0]))),
    }


def write_dataset(output_dir: Path, splits: dict[str, list[dict[str, Any]]], environments: list[dict[str, Any]], seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(output_dir / "jsonl" / f"{split}.jsonl", rows)
        write_json(output_dir / "json" / f"{split}.json", rows)

    write_json(output_dir / "environments.json", environments)
    manifest = {
        "seed": seed,
        "splits": {split: summarize_split(rows) for split, rows in splits.items()},
        "environment_count": len(environments),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def generate_all(seed: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    splits, environments = generate_id_splits(rng)
    for spec in OOD_SPECS:
        rows, envs = generate_ood_split(spec, rng)
        splits[spec.split] = rows
        environments.extend(envs)
    return splits, environments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plan.md Phase 0 datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/phase0"))
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, environments = generate_all(args.seed)
    write_dataset(args.output_dir, splits, environments, args.seed)


if __name__ == "__main__":
    main()

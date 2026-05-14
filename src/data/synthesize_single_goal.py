#!/usr/bin/env python3
"""Generate single-goal PPNL-style grid path-planning data.

Pipeline stage: standalone synthetic-data generator and compatibility helper.
Inputs: grid size, split counts, obstacle settings, and random seeds.
Outputs: raw environments, sample splits, and official-style JSON files.
Typical caller: manual data synthesis when a small independent dataset is needed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Any


ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def obstacle_key(obstacles: list[list[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((row, col) for row, col in obstacles))


def generate_environments(
    grid_size: int,
    min_obstacles: int,
    max_obstacles: int,
    num_environments: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if min_obstacles < 0 or max_obstacles < min_obstacles:
        raise ValueError("invalid obstacle range")
    if max_obstacles >= grid_size * grid_size - 1:
        raise ValueError("too many obstacles for grid size")

    environments = []
    seen = set()
    max_attempts = num_environments * 1000
    attempts = 0

    while len(environments) < num_environments and attempts < max_attempts:
        attempts += 1
        obstacle_count = rng.randint(min_obstacles, max_obstacles)
        cells = rng.sample(range(grid_size * grid_size), obstacle_count)
        obstacles = [[cell // grid_size, cell % grid_size] for cell in cells]
        key = obstacle_key(obstacles)
        if key in seen:
            continue
        seen.add(key)
        environments.append({"shape": [grid_size, grid_size], "obstacles": [list(x) for x in key]})

    if len(environments) != num_environments:
        raise RuntimeError(f"generated {len(environments)} unique environments after {attempts} attempts")
    return environments


def construct_world(grid_size: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> list[list[int]]:
    obstacle_set = {tuple(item) for item in obstacles}
    world = []
    for row in range(grid_size):
        world_row = []
        for col in range(grid_size):
            if (row, col) in obstacle_set:
                world_row.append(1)
            else:
                world_row.append(0)
        world.append(world_row)
    world[start[0]][start[1]] = 2
    world[goal[0]][goal[1]] = 3
    return world


def shortest_path_coordinates(
    grid_size: int,
    obstacles: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[list[int]] | None:
    obstacle_set = {tuple(item) for item in obstacles}
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return [[row, col] for row, col in path]

        for dr, dc in ACTION_DELTAS.values():
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                continue
            if nxt in obstacle_set or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def coordinates_to_actions(path: list[list[int]] | None) -> str:
    if path is None:
        return "Goal not reachable"

    actions = []
    for current, nxt in zip(path, path[1:]):
        dr = nxt[0] - current[0]
        dc = nxt[1] - current[1]
        for action, delta in ACTION_DELTAS.items():
            if delta == (dr, dc):
                actions.append(action)
                break
        else:
            raise ValueError(f"non-adjacent path transition: {current} -> {nxt}")
    return " ".join(actions) + (" " if actions else "")


def generate_nl(grid_size: int, obstacles: list[list[int]], start: list[int], goal: list[int]) -> str:
    if obstacles:
        obstacle_text = ""
        for index, obstacle in enumerate(obstacles):
            obstacle_text += f"({obstacle[0]},{obstacle[1]})"
            if index < len(obstacles) - 2:
                obstacle_text += ", "
            elif index == len(obstacles) - 2:
                obstacle_text += " and "
    else:
        obstacle_text = "none"
    return (
        f"You are in a {grid_size} by {grid_size} world. "
        f"There are obstacles that you have to avoid at: {obstacle_text}. "
        f"Go from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]})"
    )


def sample_worlds(
    environment: dict[str, Any],
    trials: int,
    rng: random.Random,
    require_reachable: bool,
) -> list[dict[str, Any]]:
    grid_size = int(environment["shape"][0])
    obstacles = [list(item) for item in environment["obstacles"]]
    obstacle_set = {tuple(item) for item in obstacles}
    free_cells = [(row, col) for row in range(grid_size) for col in range(grid_size) if (row, col) not in obstacle_set]
    if len(free_cells) < 2:
        return []

    worlds = []
    seen_pairs = set()
    max_attempts = max(1000, trials * 100)
    attempts = 0

    while len(worlds) < trials and attempts < max_attempts:
        attempts += 1
        start, goal = rng.sample(free_cells, 2)
        if (start, goal) in seen_pairs:
            continue
        path = shortest_path_coordinates(grid_size, obstacles, start, goal)
        if require_reachable and path is None:
            continue
        seen_pairs.add((start, goal))
        start_list = [start[0], start[1]]
        goal_list = [goal[0], goal[1]]
        worlds.append(
            {
                "world": construct_world(grid_size, obstacles, start_list, goal_list),
                "obstacles": obstacles,
                "start": start_list,
                "goals": [goal_list],
            }
        )
    return worlds


def world_to_sample(world: dict[str, Any]) -> dict[str, Any]:
    grid_size = len(world["world"])
    start = world["start"]
    goal = world["goals"][0]
    coordinates = shortest_path_coordinates(grid_size, world["obstacles"], tuple(start), tuple(goal))
    actions = coordinates_to_actions(coordinates)
    return {
        "world": world["world"],
        "nl_description": generate_nl(grid_size, world["obstacles"], start, goal),
        "solution_coordinates": coordinates if coordinates is not None else "Goal not reachable",
        "agent_as_a_point": actions,
        "agent_has_direction": "",
    }


def split_seen_worlds(worlds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(worlds) < 3:
        return worlds, [], []
    train_end = int(0.8 * len(worlds))
    dev_count = max(1, int(0.1 * len(worlds)))
    dev_end = min(len(worlds), train_end + dev_count)
    return worlds[:train_end], worlds[train_end:dev_end], worlds[dev_end:]


def grid_tag(grid_size: int, setting_name: str) -> str:
    if setting_name:
        return f"{grid_size}x{grid_size}{setting_name}"
    return f"{grid_size}x{grid_size}"


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    environments = generate_environments(
        args.grid_size,
        args.min_obstacles,
        args.max_obstacles,
        args.num_environments,
        rng,
    )

    env_train_end = int(args.train_env_fraction * len(environments))
    seen_envs = environments[:env_train_end]
    unseen_envs = environments[env_train_end:]

    train_worlds: list[dict[str, Any]] = []
    dev_worlds: list[dict[str, Any]] = []
    test_seen_worlds: list[dict[str, Any]] = []
    test_unseen_worlds: list[dict[str, Any]] = []

    for environment in seen_envs:
        worlds = sample_worlds(environment, args.trials_per_environment, rng, args.require_reachable)
        train_part, dev_part, test_part = split_seen_worlds(worlds)
        train_worlds.extend(train_part)
        dev_worlds.extend(dev_part)
        test_seen_worlds.extend(test_part)

    for environment in unseen_envs:
        test_unseen_worlds.extend(sample_worlds(environment, args.trials_per_environment, rng, args.require_reachable))

    samples = {
        "train": [world_to_sample(world) for world in train_worlds],
        "dev": [world_to_sample(world) for world in dev_worlds],
        "test_seen": [world_to_sample(world) for world in test_seen_worlds],
        "test_unseen": [world_to_sample(world) for world in test_unseen_worlds],
    }

    return {
        "environments": environments,
        "worlds": {
            "train": train_worlds,
            "dev": dev_worlds,
            "test_seen": test_seen_worlds,
            "test_unseen": test_unseen_worlds,
        },
        "samples": samples,
        "manifest": {
            "seed": args.seed,
            "grid_size": args.grid_size,
            "min_obstacles": args.min_obstacles,
            "max_obstacles": args.max_obstacles,
            "num_environments": args.num_environments,
            "train_env_fraction": args.train_env_fraction,
            "trials_per_environment": args.trials_per_environment,
            "require_reachable": args.require_reachable,
            "counts": {
                "environments": len(environments),
                "train": len(samples["train"]),
                "dev": len(samples["dev"]),
                "test_seen": len(samples["test_seen"]),
                "test_unseen": len(samples["test_unseen"]),
            },
        },
    }


def write_dataset(output_dir: Path, tag: str, data: dict[str, Any]) -> None:
    write_json(output_dir / "manifest.json", data["manifest"])
    write_json(output_dir / "raw_environments.json", data["environments"])

    worlds_dir = output_dir / "worlds"
    samples_dir = output_dir / "samples"
    for split, rows in data["worlds"].items():
        write_json(worlds_dir / f"{split}.json", rows)
    for split, rows in data["samples"].items():
        write_json(samples_dir / f"{split}.json", rows)

    official_dir = output_dir / "official_style"
    write_json(official_dir / f"1_train_set_{tag}_samples.json", data["samples"]["train"])
    write_json(official_dir / f"1dev_set_{tag}_samples.json", data["samples"]["dev"])
    write_json(official_dir / f"1_goals_test_seen_{tag}_samples.json", data["samples"]["test_seen"])
    write_json(official_dir / f"1goals_unseen_{tag}_samples.json", data["samples"]["test_unseen"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-goal PPNL-style datasets.")
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--min-obstacles", type=int, required=True)
    parser.add_argument("--max-obstacles", type=int, required=True)
    parser.add_argument("--num-environments", type=int, required=True)
    parser.add_argument("--trials-per-environment", type=int, default=30)
    parser.add_argument("--train-env-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--setting-name", default="", help="Optional suffix such as dense or more_obstacles.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-unreachable",
        action="store_true",
        help="Keep start/goal pairs even when no path exists. By default only reachable rows are emitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_env_fraction < 1.0):
        raise ValueError("--train-env-fraction must be between 0 and 1")
    args.require_reachable = not args.allow_unreachable

    data = generate_dataset(args)
    tag = grid_tag(args.grid_size, args.setting_name)
    write_dataset(args.output_dir, tag, data)
    print(json.dumps(data["manifest"], indent=2))


if __name__ == "__main__":
    main()

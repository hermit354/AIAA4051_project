#!/usr/bin/env python3
"""Generate compact dense large-grid held-out splits for rollout stress tests.

Pipeline stage: data generation for size and density generalization experiments.
Inputs: command-line size/count parameters and deterministic random seeds.
Outputs: JSON/JSONL datasets plus environment manifests under a generated-data directory.
Typical caller: scripts that prepare Phase 2 size-generalization evaluation data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_phase0 import generate_pairs, make_example
from src.data.make_sft_method_datasets import convert_react_trace
from src.data.ppnl_io import write_json, write_jsonl


DEFAULT_SIZES = (12, 14, 16)


def obstacle_key(obstacles: list[list[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(row), int(col)) for row, col in obstacles))


def make_environment(
    size: int,
    obstacle_count: int,
    env_index: int,
    rng: random.Random,
    seen: set[tuple[int, tuple[tuple[int, int], ...]]],
    prefix: str,
) -> dict[str, Any] | None:
    cells = list(range(size * size))
    for _ in range(3000):
        sampled = rng.sample(cells, obstacle_count)
        obstacles = [[cell // size, cell % size] for cell in sampled]
        key = obstacle_key(obstacles)
        scoped = (obstacle_count, key)
        if scoped in seen:
            continue
        seen.add(scoped)
        return {
            "environment_id": f"{size}x{size}_obs{obstacle_count}_{prefix}{env_index:03d}",
            "grid_size": [size, size],
            "grid_shape": f"{size}x{size}",
            "obstacle_count": obstacle_count,
            "obstacles": [list(item) for item in key],
        }
    return None


def path_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shortest = [int(row["shortest_path_length"]) for row in rows]
    manhattan = [int(row["manhattan_distance"]) for row in rows]
    detours = [s - m for s, m in zip(shortest, manhattan)]
    ratios = [s / max(1, m) for s, m in zip(shortest, manhattan)]
    return {
        "examples": len(rows),
        "obstacle_count": sorted(set(int(row["obstacle_count"]) for row in rows)),
        "shortest": {
            "min": min(shortest),
            "max": max(shortest),
            "mean": sum(shortest) / len(shortest),
        },
        "manhattan": {
            "min": min(manhattan),
            "max": max(manhattan),
            "mean": sum(manhattan) / len(manhattan),
        },
        "detour": {
            "mean": sum(detours) / len(detours),
            "fraction_positive": sum(1 for item in detours if item > 0) / len(detours),
            "fraction_ge_2": sum(1 for item in detours if item >= 2) / len(detours),
        },
        "shortest_over_manhattan": {
            "mean": sum(ratios) / len(ratios),
            "max": max(ratios),
        },
    }


def generate_split(
    size: int,
    examples: int,
    env_count: int,
    rng: random.Random,
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs_per_env = max(1, examples // env_count)
    obstacle_counts = [
        max(1, round(size * size * 0.20)),
        max(1, round(size * size * 0.25)),
        max(1, round(size * size * 0.30)),
    ]
    rows: list[dict[str, Any]] = []
    envs: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    attempts = 0
    env_index = 0
    while len(rows) < examples and attempts < env_count * 200:
        attempts += 1
        obstacle_count = obstacle_counts[env_index % len(obstacle_counts)]
        env = make_environment(size, obstacle_count, env_index, rng, seen, prefix)
        env_index += 1
        if env is None:
            continue
        try:
            pairs = generate_pairs(env, pairs_per_env, rng)
        except RuntimeError:
            continue
        envs.append(env)
        split = f"dense_large_{size}x{size}"
        for pair_index, (start, goal, path) in enumerate(pairs):
            rows.append(make_example(env, split, pair_index, start, goal, path, is_dense=True))
            if len(rows) >= examples:
                break
    if len(rows) != examples:
        raise RuntimeError(f"generated {len(rows)}/{examples} examples for {size}x{size}")
    return rows, envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense large-grid held-out splits.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/phase2_size_generalization/dense_large_small"))
    parser.add_argument("--seed", type=int, default=9059)
    parser.add_argument("--examples-per-size", type=int, default=300)
    parser.add_argument("--envs-per-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    all_envs: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "examples_per_size": args.examples_per_size,
        "envs_per_size_requested": args.envs_per_size,
        "splits": {},
    }
    for size in DEFAULT_SIZES:
        split = f"dense_large_{size}x{size}"
        rows, envs = generate_split(size, args.examples_per_size, args.envs_per_size, rng, split)
        converted = convert_react_trace(rows, "react_trace_compact")
        write_jsonl(args.output_dir / "jsonl" / f"{split}.jsonl", converted)
        write_json(args.output_dir / "json" / f"{split}.json", converted)
        all_envs.extend(envs)
        manifest["splits"][split] = path_stats(rows)
    write_json(args.output_dir / "environments.json", all_envs)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

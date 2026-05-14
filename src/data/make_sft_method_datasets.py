#!/usr/bin/env python3
"""Materialize Phase 2 datasets for SFT-method ablations.

Pipeline stage: dataset conversion before training method variants.
Inputs: base JSONL records and the requested supervision format.
Outputs: per-method JSONL splits and a manifest.
Typical caller: scripts/run_phase2_sft_methods.sh and related launchers.
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

from src.data.ppnl_io import load_records, write_json, write_jsonl


DEFAULT_SPLITS = [
    "train",
    "val",
    "test_unseen_placement",
    "test_unseen_environment",
    "ood_size_5x5",
    "ood_size_7x7",
    "ood_size_8x8",
    "ood_size_9x9",
    "ood_size_10x10",
    "ood_dense_6x6",
    "ood_dense_5x5",
    "ood_dense_7x7",
    "ood_aspect_10x5",
    "ood_aspect_6x9",
]

METHODS = ("coordinate_trace", "multi_template", "react_trace_arrow", "react_trace_compact", "react_trace_minimal")


def obstacle_text(sample: dict[str, Any]) -> str:
    """Format obstacle coordinates for natural-language templates."""
    obstacles = sample.get("obstacles", [])
    if not obstacles:
        return "none"
    return ", ".join(f"({row},{col})" for row, col in obstacles)


def template_question(sample: dict[str, Any], template_id: int) -> str:
    """Return one of several input templates for multi-template SFT."""
    rows, cols = sample["grid_size"]
    start = sample["start"]
    goal = sample["goal"]
    obstacles = obstacle_text(sample)
    templates = [
        (
            f"Grid: {rows} by {cols}. Start at ({start[0]},{start[1]}). "
            f"Goal at ({goal[0]},{goal[1]}). Obstacles: {obstacles}."
        ),
        (
            f"Find the shortest route on a {rows}x{cols} grid from "
            f"({start[0]},{start[1]}) to ({goal[0]},{goal[1]}). "
            f"Blocked cells are {obstacles}. Return only actions."
        ),
        (
            f"Rows={rows}; Cols={cols}; Start=({start[0]},{start[1]}); "
            f"Goal=({goal[0]},{goal[1]}); Obstacles={obstacles}. "
            "Use up, down, left, right."
        ),
    ]
    return templates[template_id % len(templates)]


def coordinate_trace_target(sample: dict[str, Any]) -> str:
    """Create a verbose coordinate-by-coordinate supervision target."""
    coords = sample.get("solution_coordinates")
    actions = str(sample.get("target", "")).split()
    if not isinstance(coords, list) or len(coords) != len(actions) + 1:
        raise ValueError(f"{sample.get('example_id', sample.get('id'))}: invalid coordinate/action trace")

    parts = []
    for step, action in enumerate(actions):
        row, col = coords[step]
        next_row, next_col = coords[step + 1]
        parts.append(f"at ({row},{col}) take {action} to ({next_row},{next_col})")
    final_row, final_col = coords[-1]
    parts.append(f"stop at ({final_row},{final_col})")
    return "; ".join(parts)


def final_actions(sample: dict[str, Any]) -> str:
    """Return the normalized final action sequence for a record."""
    return " ".join(str(sample.get("target", sample.get("agent_as_a_point", ""))).split())


def react_trace_arrow_target(sample: dict[str, Any]) -> str:
    """Create a trace target that shows coordinate transitions with arrows."""
    coords = sample.get("solution_coordinates")
    actions = str(sample.get("target", "")).split()
    if not isinstance(coords, list) or len(coords) != len(actions) + 1:
        raise ValueError(f"{sample.get('example_id', sample.get('id'))}: invalid coordinate/action trace")
    parts = []
    for step, action in enumerate(actions):
        row, col = coords[step]
        next_row, next_col = coords[step + 1]
        parts.append(f"({row},{col}) {action} -> ({next_row},{next_col})")
    parts.append(f"FINAL: {final_actions(sample)}")
    return "; ".join(parts)


def react_trace_compact_target(sample: dict[str, Any]) -> str:
    """Create a compact trace target followed by final actions."""
    coords = sample.get("solution_coordinates")
    actions = str(sample.get("target", "")).split()
    if not isinstance(coords, list) or len(coords) != len(actions) + 1:
        raise ValueError(f"{sample.get('example_id', sample.get('id'))}: invalid coordinate/action trace")
    start_row, start_col = coords[0]
    parts = [f"S=({start_row},{start_col})"]
    for step, action in enumerate(actions):
        next_row, next_col = coords[step + 1]
        parts.append(f"{action}:({next_row},{next_col})")
    parts.append(f"FINAL {final_actions(sample)}")
    return "; ".join(parts)


ACTION_ABBREVIATIONS = {"up": "U", "down": "D", "left": "L", "right": "R"}


def react_trace_minimal_target(sample: dict[str, Any]) -> str:
    """Create the shortest coordinate/action trace variant."""
    coords = sample.get("solution_coordinates")
    actions = str(sample.get("target", "")).split()
    if not isinstance(coords, list) or len(coords) != len(actions) + 1:
        raise ValueError(f"{sample.get('example_id', sample.get('id'))}: invalid coordinate/action trace")
    parts = []
    for step, action in enumerate(actions):
        row, col = coords[step]
        parts.append(f"({row},{col}) {ACTION_ABBREVIATIONS[action]}")
    final_row, final_col = coords[-1]
    parts.append(f"({final_row},{final_col}) STOP")
    return " ".join(parts)


def convert_coordinate_trace(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert records into coordinate-trace SFT examples."""
    converted = []
    for record in records:
        row = dict(record)
        row["sft_method"] = "coordinate_trace"
        row["question"] = (
            f"{row.get('natural_input', row.get('question', ''))} "
            "Return a coordinate trace, naming each current position, action, and next position."
        ).strip()
        row["nl_description"] = row["question"]
        row["target"] = coordinate_trace_target(row)
        converted.append(row)
    return converted


def convert_react_trace(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    """Convert records into one of the ReAct-style trace formats."""
    converted = []
    target_fns = {
        "react_trace_arrow": react_trace_arrow_target,
        "react_trace_compact": react_trace_compact_target,
        "react_trace_minimal": react_trace_minimal_target,
    }
    target_fn = target_fns[method]
    for record in records:
        row = dict(record)
        row["sft_method"] = method
        if method == "react_trace_minimal":
            instruction = "Return only a compact trace using U/D/L/R between coordinates and end with STOP."
        else:
            instruction = "Return a concise move trace ending with FINAL and the action sequence."
        row["question"] = f"{row.get('natural_input', row.get('question', ''))} {instruction}".strip()
        row["nl_description"] = row["question"]
        row["target"] = target_fn(row)
        converted.append(row)
    return converted


def convert_multi_template(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Create template-augmented examples, expanding only the training split."""
    converted = []
    template_ids = range(3) if split == "train" else range(1)
    for record in records:
        for template_id in template_ids:
            row = dict(record)
            row["sft_method"] = "multi_template"
            row["template_id"] = template_id
            row["question"] = template_question(row, template_id)
            row["nl_description"] = row["question"]
            if split == "train":
                row["id"] = f"{row.get('id', row.get('example_id'))}_tpl{template_id}"
                row["example_id"] = f"{row.get('example_id', row.get('id'))}_tpl{template_id}"
            converted.append(row)
    return converted


def convert_records(records: list[dict[str, Any]], split: str, method: str) -> list[dict[str, Any]]:
    """Dispatch one split through the requested SFT conversion method."""
    if method == "coordinate_trace":
        return convert_coordinate_trace(records)
    if method == "multi_template":
        return convert_multi_template(records, split)
    if method in {"react_trace_arrow", "react_trace_compact", "react_trace_minimal"}:
        return convert_react_trace(records, method)
    raise ValueError(f"unknown method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2 SFT-method datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/generated/phase2_sft_method"))
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {"source_dir": str(args.source_dir), "methods": {}, "splits": args.splits}
    for method in args.methods:
        method_root = args.output_root / method / "jsonl"
        counts = {}
        for split in args.splits:
            source_path = args.source_dir / f"{split}.jsonl"
            converted = convert_records(load_records(source_path), split, method)
            write_jsonl(method_root / f"{split}.jsonl", converted)
            counts[split] = len(converted)
        manifest["methods"][method] = {"root": str(args.output_root / method), "counts": counts}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize input-representation ablation datasets from Phase 0 records.

Pipeline stage: dataset conversion for input-format experiments.
Inputs: Phase 0 JSONL records containing natural, structured, and grid-matrix fields.
Outputs: per-format JSONL splits and a manifest.
Typical caller: scripts/run_phase2_input_format.sh.
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

FORMATS = ("natural", "structured", "grid_matrix", "hybrid")


def obstacles_text(sample: dict[str, Any]) -> str:
    obstacles = sample.get("obstacles", [])
    if not obstacles:
        return "none"
    return ", ".join(f"({row},{col})" for row, col in obstacles)


def format_question(sample: dict[str, Any], input_format: str) -> str:
    if input_format == "natural":
        return str(sample.get("nl_description", sample.get("natural_input", sample.get("question", "")))).strip()
    if input_format == "structured":
        return str(sample["structured_input"]).strip()
    if input_format == "grid_matrix":
        return (
            "Find a shortest path from S to G. "
            "Use only up, down, left, right. X cells are obstacles.\n"
            f"{sample['grid_matrix_input']}"
        )
    if input_format == "hybrid":
        rows, cols = sample["grid_size"]
        start = sample["start"]
        goal = sample["goal"]
        return (
            f"Grid size: {rows} by {cols}. Start: ({start[0]},{start[1]}). "
            f"Goal: ({goal[0]},{goal[1]}). Obstacles: {obstacles_text(sample)}.\n"
            "Matrix with S=start, G=goal, X=obstacle, .=empty:\n"
            f"{sample['grid_matrix_input']}\n"
            "Return only actions: up, down, left, right."
        )
    raise ValueError(f"unknown input format: {input_format}")


def convert_records(records: list[dict[str, Any]], input_format: str) -> list[dict[str, Any]]:
    converted = []
    for record in records:
        row = dict(record)
        question = format_question(row, input_format)
        row["input_format"] = input_format
        row["question"] = question
        row["nl_description"] = question
        converted.append(row)
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2 input-format datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/generated/phase2_input_format"))
    parser.add_argument("--formats", nargs="*", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {"source_dir": str(args.source_dir), "formats": {}, "splits": args.splits}
    for input_format in args.formats:
        format_root = args.output_root / input_format / "jsonl"
        counts = {}
        for split in args.splits:
            source_path = args.source_dir / f"{split}.jsonl"
            records = load_records(source_path)
            converted = convert_records(records, input_format)
            write_jsonl(format_root / f"{split}.jsonl", converted)
            counts[split] = len(converted)
        manifest["formats"][input_format] = {"root": str(args.output_root / input_format), "counts": counts}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Shared I/O utilities for single-goal PPNL-style path-planning data.

Pipeline stage: common support code used by data, model, and evaluation modules.
Inputs: JSON/JSONL dataset files, prediction files, and generic Python data structures.
Outputs: normalized records, seq2seq examples, and JSON/JSONL artifacts.
Typical caller: nearly every command-line entry point in src/data, src/models, and src/eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load path-planning records from JSON, JSONL, or wrapped JSON objects."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        return load_jsonl(input_path)

    data = load_json(input_path)
    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "records" in data:
            data = data["records"]
        elif "samples" in data and isinstance(data["samples"], list):
            data = data["samples"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records")
    return data


def normalize_action_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return " ".join(str(value).strip().lower().split())


def is_reachable_target(target: str) -> bool:
    return "goal not reachable" not in target.lower()


def sample_to_seq2seq(sample: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Convert either a PPNL sample or converted question/target row.

    Returns None for unreachable or malformed targets so training scripts can
    skip records that are not valid supervised path-planning examples.
    """
    question = sample.get("question", sample.get("nl_description", sample.get("english", "")))
    target = sample.get("target", sample.get("agent_as_a_point", sample.get("ground_truth", "")))
    target = normalize_action_text(target)

    if not question or not target or not is_reachable_target(target):
        return None

    return {
        "id": sample.get("id", index),
        "question": str(question).strip(),
        "target": target,
    }


def load_seq2seq_examples(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load records and normalize them into seq2seq question/target examples."""
    examples = []
    for index, sample in enumerate(load_records(path)):
        converted = sample_to_seq2seq(sample, index)
        if converted is None:
            continue
        examples.append(converted)
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def load_original_samples(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    data = load_records(path)
    if max_samples is not None:
        return data[:max_samples]
    return data

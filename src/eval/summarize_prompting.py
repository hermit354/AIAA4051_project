#!/usr/bin/env python3
"""Summarize prompt-engineering metrics by ID and OOD groups.

Pipeline stage: result aggregation after local or API prompting sweeps.
Inputs: prompting output directories with per-style and per-split metrics.
Outputs: CSV summaries for prompt-method comparisons.
Typical caller: prompt experiment analysis workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.summarize_phase1 import GROUPS
from src.models.model_registry import model_names


PROMPTS = ["zero_shot", "one_shot", "few_shot", "cot", "react", "structured"]


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def weighted_average(items: list[dict[str, Any]], key: str) -> float:
    total = sum(item["total"] for item in items)
    return sum(item[key] * item["total"] for item in items) / total


def summarize_prompt(root: Path, model: str, prompt: str) -> list[dict[str, Any]]:
    rows = []
    prompt_dir = root / model / prompt
    for group, splits in GROUPS.items():
        paths = [prompt_dir / split / "metrics.json" for split in splits]
        if not all(path.exists() for path in paths):
            continue
        metrics = [load_metrics(path) for path in paths]
        rows.append(
            {
                "model": model,
                "prompt_type": prompt,
                "group": group,
                "splits": "+".join(splits),
                "total": sum(item["total"] for item in metrics),
                "success_rate": weighted_average(metrics, "success_rate"),
                "feasibility": weighted_average(metrics, "feasibility"),
                "optimality": weighted_average(metrics, "optimality"),
                "exact_match": weighted_average(metrics, "exact_match"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prompt-engineering metrics.")
    parser.add_argument("--root", type=Path, default=Path("outputs/phase2_prompting"))
    parser.add_argument("--models", nargs="*", default=model_names())
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_prompting/tables/prompting_group_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for model in args.models:
        for prompt in args.prompts:
            rows.extend(summarize_prompt(args.root, model, prompt))
    if not rows:
        raise SystemExit(f"no prompt metrics found under {args.root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

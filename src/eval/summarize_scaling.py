#!/usr/bin/env python3
"""Summarize Phase 2.5 data-scaling metrics.

Pipeline stage: result aggregation for training-data scaling studies.
Inputs: per-percentage output roots and optional Phase 1 reference summaries.
Outputs: CSV tables that compare training size with evaluation performance.
Typical caller: scripts/run_phase2_data_scaling.sh follow-up analysis.
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


def load_json(path: Path) -> dict[str, Any]:
    return json.load(path.open())


def weighted_average(items: list[dict[str, Any]], key: str) -> float:
    total = sum(item["total"] for item in items)
    return sum(item[key] * item["total"] for item in items) / max(1, total)


def summarize_run(run_dir: Path, label_fields: dict[str, Any]) -> list[dict[str, Any]]:
    train_summary = load_json(run_dir / "train_summary.json") if (run_dir / "train_summary.json").exists() else {}
    rows = []
    for group, splits in GROUPS.items():
        paths = [run_dir / split / "metrics.json" for split in splits]
        if not all(path.exists() for path in paths):
            continue
        metrics = [load_json(path) for path in paths]
        rows.append(
            {
                **label_fields,
                "group": group,
                "splits": "+".join(splits),
                "total": sum(item["total"] for item in metrics),
                "best_val_exact": train_summary.get("best_exact_match"),
                "best_step": train_summary.get("best_step"),
                "success_rate": weighted_average(metrics, "success_rate"),
                "feasibility": weighted_average(metrics, "feasibility"),
                "optimality": weighted_average(metrics, "optimality"),
                "exact_match": weighted_average(metrics, "exact_match"),
            }
        )
    return rows


def load_phase1_model_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            tier = "small" if model.endswith("small") else "base"
            rows.append(
                {
                    "analysis": "model_scaling",
                    "tier": tier,
                    "model": model,
                    "group": row["group"],
                    "splits": row["splits"],
                    "total": row["total"],
                    "best_val_exact": row.get("best_val_exact"),
                    "best_step": "",
                    "success_rate": row["success_rate"],
                    "feasibility": row["feasibility"],
                    "optimality": row["optimality"],
                    "exact_match": row["exact_match"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scaling experiments.")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/phase2_scaling/data_scaling"))
    parser.add_argument("--model", default="flan-t5-base")
    parser.add_argument("--percents", nargs="*", type=int, default=[1, 5, 10, 25, 50])
    parser.add_argument("--phase1-summary", type=Path, default=Path("outputs/tables/table1_phase1_group_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_scaling/tables/scaling_group_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for percent in args.percents:
        rows.extend(
            summarize_run(
                args.data_root / f"p{percent}" / args.model,
                {"analysis": "data_scaling", "tier": "", "model": args.model, "train_percent": percent},
            )
        )
    if args.phase1_summary.exists():
        with args.phase1_summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model"] == args.model:
                    rows.append(
                        {
                            "analysis": "data_scaling",
                            "tier": "",
                            "model": args.model,
                            "train_percent": 100,
                            "group": row["group"],
                            "splits": row["splits"],
                            "total": row["total"],
                            "best_val_exact": row.get("best_val_exact"),
                            "best_step": "",
                            "success_rate": row["success_rate"],
                            "feasibility": row["feasibility"],
                            "optimality": row["optimality"],
                            "exact_match": row["exact_match"],
                        }
                    )
        rows.extend(load_phase1_model_rows(args.phase1_summary))
    if not rows:
        raise SystemExit("no scaling rows found")
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "analysis",
        "tier",
        "model",
        "train_percent",
        "group",
        "splits",
        "total",
        "best_val_exact",
        "best_step",
        "success_rate",
        "feasibility",
        "optimality",
        "exact_match",
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

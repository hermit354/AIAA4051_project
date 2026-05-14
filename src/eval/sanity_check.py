#!/usr/bin/env python3
"""Run controlled sanity checks for the path-planning evaluator.

Pipeline stage: lightweight verification for evaluation logic.
Inputs: built-in toy records that cover success and failure cases.
Outputs: metrics JSON and evaluated trace JSONL files.
Typical caller: README smoke tests and pre-commit validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_predictions import evaluate_records, load_table, write_json, write_jsonl


WORLD = [
    [2, 0, 3, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]


def case(case_id: str, generated: str, expected: dict[str, object]) -> dict[str, object]:
    return {
        "id": case_id,
        "world": WORLD,
        "english": (
            "You are in a 4 by 4 world. There are obstacles that you have to avoid at: "
            "(1,1). Go from (0,0) to (0,2)"
        ),
        "ground_truth": "right right ",
        "generated": generated,
        "expected": expected,
    }


def controlled_cases() -> list[dict[str, object]]:
    return [
        case("oracle_shortest_path", "right right ", {"success": True, "feasible": True, "optimal": True}),
        case("valid_non_optimal_path", "down down right right up up ", {"success": True, "feasible": True, "optimal": False}),
        case("out_of_bounds_path", "left ", {"success": False, "feasible": False, "failure_type": "OUT_OF_BOUNDS"}),
        case("obstacle_collision", "down right ", {"success": False, "feasible": False, "failure_type": "HIT_OBSTACLE"}),
        case(
            "invalid_action_token",
            "right jump ",
            {"success": False, "feasible": False, "failure_type": "INVALID_TOKEN"},
        ),
        case(
            "early_stop_before_goal",
            "right ",
            {"success": False, "feasible": True, "optimal": False, "failure_type": "PREMATURE_STOP"},
        ),
    ]


def assert_expected(traces: list[dict[str, object]]) -> list[str]:
    failures = []
    for trace in traces:
        expected = next(item["expected"] for item in controlled_cases() if item["id"] == trace["id"])
        for key, value in expected.items():
            if trace.get(key) != value:
                failures.append(f"{trace['id']}: expected {key}={value}, got {trace.get(key)}")
    return failures


def official_oracle_metrics(dataset_path: Path) -> dict[str, object]:
    dataset = load_table(dataset_path)
    records = []
    skipped_unreachable = 0
    for index, item in enumerate(dataset):
        target = item.get("agent_as_a_point", "")
        if "Goal not reachable" in target:
            skipped_unreachable += 1
            continue
        records.append(
            {
                **item,
                "id": f"official_oracle_{index}",
                "english": item.get("nl_description", ""),
                "ground_truth": target,
                "generated": target,
            }
        )
    metrics, _ = evaluate_records(records)
    metrics["dataset"] = str(dataset_path)
    metrics["skipped_unreachable"] = skipped_unreachable
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-0 evaluator sanity checks.")
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/metrics/evaluator_sanity_check.json"))
    parser.add_argument("--traces-out", type=Path, default=Path("outputs/predictions/evaluator_sanity_cases.jsonl"))
    parser.add_argument(
        "--oracle-dataset",
        type=Path,
        default=Path(
            "data/raw/ppnl_single_goal/"
            "1_goals_test_seen_6x6_samples.json"
        ),
    )
    args = parser.parse_args()

    records = controlled_cases()
    controlled_metrics, traces = evaluate_records(records)
    failures = assert_expected(traces)
    oracle_metrics = official_oracle_metrics(args.oracle_dataset)

    output = {
        "controlled_sanity": controlled_metrics,
        "official_oracle": oracle_metrics,
        "passed": not failures
        and oracle_metrics["success_rate"] == 1.0
        and oracle_metrics["feasibility"] == 1.0
        and oracle_metrics["optimality"] == 1.0,
        "failures": failures,
    }

    for trace, record in zip(traces, records):
        trace["expected"] = record["expected"]

    write_json(args.metrics_out, output)
    write_jsonl(args.traces_out, traces)
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

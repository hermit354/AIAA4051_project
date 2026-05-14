#!/usr/bin/env python3
"""Validate that local baseline model directories can be loaded.

Pipeline stage: preflight check before Phase 1 baseline training.
Inputs: model names from the local registry and source_models directories.
Outputs: JSON integrity reports describing which models load successfully.
Typical caller: scripts/run_phase1_baselines_gpu5.sh.
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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.ppnl_io import write_json
from src.models.model_registry import model_names, model_path


def check_model(name: str) -> dict[str, Any]:
    path = model_path(name)
    result: dict[str, Any] = {"name": name, "path": str(path), "exists": path.exists(), "ok": False}
    if not path.exists():
        result["error"] = "path does not exist"
        return result
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
        result.update(
            {
                "ok": True,
                "tokenizer": type(tokenizer).__name__,
                "model": type(model).__name__,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        del model
        del tokenizer
    except Exception as exc:  # noqa: BLE001 - report actionable local model failures.
        result["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local Phase 1 model directories.")
    parser.add_argument("--models", nargs="*", default=model_names())
    parser.add_argument("--report-out", type=Path, default=Path("outputs/local_model_integrity.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [check_model(name) for name in args.models]
    report = {"passed": all(item["ok"] for item in results), "models": results}
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report whether a sufficiently free GPU is available.

Pipeline stage: preflight check before expensive training or inference runs.
Inputs: nvidia-smi output and command-line utilization thresholds.
Outputs: JSON compute reports and optional nonzero exit status when no GPU is free.
Typical caller: shell launchers before starting model jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import write_json


def nvidia_smi_rows() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001 - environment report should include exact failure.
        return [], f"{type(exc).__name__}: {exc}"

    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, name, memory_used, memory_total, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": int(memory_used),
                "memory_total_mb": int(memory_total),
                "utilization_gpu_percent": int(utilization),
            }
        )
    return rows, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GPU availability without killing or modifying jobs.")
    parser.add_argument("--max-used-mb", type=int, default=500)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--report-out", type=Path, default=Path("outputs/compute_check.json"))
    parser.add_argument("--require-free-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus, nvidia_error = nvidia_smi_rows()
    free_gpus = [
        gpu
        for gpu in gpus
        if gpu["memory_used_mb"] <= args.max_used_mb and gpu["utilization_gpu_percent"] <= args.max_utilization
    ]
    report = {
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi_error": nvidia_error,
        "gpus": gpus,
        "free_gpus": free_gpus,
        "has_fully_free_gpu": bool(free_gpus),
        "policy": {
            "max_used_mb": args.max_used_mb,
            "max_utilization": args.max_utilization,
            "no_job_killing": True,
        },
    }
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2))
    if args.require_free_gpu and not free_gpus:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

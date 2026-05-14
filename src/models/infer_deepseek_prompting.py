#!/usr/bin/env python3
"""Run DeepSeek API prompt-only path-planning inference.

Pipeline stage: external API prompting baseline.
Inputs: dataset records, prompt style, API key environment variable, and retry settings.
Outputs: prediction JSONL files with API metadata and optional partial progress files.
Typical caller: scripts/run_phase2_deepseek_prompting.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_original_samples, write_jsonl
from src.models.infer_prompting import build_prompt, extract_actions


PROMPT_TYPES = ["zero_shot", "one_shot", "few_shot", "cot", "react", "structured"]
DIRECT_PROMPTS = {"zero_shot", "one_shot", "few_shot", "structured"}


def grid_dimensions(sample: dict[str, Any]) -> tuple[int, int] | None:
    """Infer grid dimensions from any supported dataset schema."""
    world = sample.get("world")
    if isinstance(world, list) and world and isinstance(world[0], list):
        return len(world), len(world[0])
    grid_shape = sample.get("grid_shape")
    if isinstance(grid_shape, str) and "x" in grid_shape:
        left, right = grid_shape.lower().split("x", 1)
        return int(left.strip()), int(right.strip())
    if isinstance(grid_shape, list):
        values = [str(part) for part in grid_shape if str(part).isdigit()]
        if len(values) >= 2:
            return int(values[0]), int(values[1])
    grid_size = sample.get("grid_size")
    if isinstance(grid_size, int):
        return grid_size, grid_size
    if isinstance(grid_size, str) and grid_size.isdigit():
        size = int(grid_size)
        return size, size
    return None


def action_budget(sample: dict[str, Any], floor: int = 12) -> int:
    """Compute a conservative action cap from the visible grid size."""
    dims = grid_dimensions(sample)
    if dims is None:
        return floor
    rows, cols = dims
    # The Phase 0/OOD generator uses relatively short shortest paths. This cap
    # is based only on visible grid dimensions and prevents API models from
    # spending hundreds of tokens on loops.
    return max(floor, rows + cols + 8)


def api_prompt(sample: dict[str, Any], prompt_type: str, prompt: str) -> str:
    """Add API-specific output guardrails to a base prompt."""
    budget = action_budget(sample)
    guardrail = (
        f"\nOutput constraints: produce at most {budget} actions. "
        "Stop immediately after the last action. Do not add commentary. "
        "Do not repeat cycles such as up down, down up, left right, or right left."
    )
    if prompt_type in DIRECT_PROMPTS:
        return prompt.rstrip() + guardrail
    if prompt_type == "react":
        return (
            prompt.rstrip()
            + guardrail
            + "\nUse compact notes only if necessary, and always finish with Final answer: <actions>."
        )
    if prompt_type == "cot":
        return (
            prompt.rstrip()
            + guardrail
            + "\nKeep the reasoning concise and always finish with Final answer: <actions>."
        )
    return prompt


def max_tokens_for_sample(args: argparse.Namespace, sample: dict[str, Any]) -> int:
    """Choose a token budget that matches the prompt style and grid size."""
    if args.prompt_type in DIRECT_PROMPTS:
        return min(args.max_new_tokens, action_budget(sample) + 16)
    if args.prompt_type == "react":
        return min(args.max_new_tokens, max(96, action_budget(sample) * 4))
    return args.max_new_tokens


def prediction_row(
    sample: dict[str, Any],
    index: int,
    prompt: str,
    prompt_type: str,
    raw_output: str,
    model: str,
    api_error: str | None = None,
    api_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one raw API response into the common prediction schema.

    The raw response and raw extracted actions are preserved for debugging,
    while `generated` and `model_output` are capped to an executable budget for
    downstream evaluation.
    """
    raw_actions = extract_actions(raw_output)
    action_limit = action_budget(sample)
    action_parts = raw_actions.split()
    actions = " ".join(action_parts[:action_limit])
    row = {
        "id": sample.get("id", sample.get("example_id", index)),
        "english": sample.get("nl_description", sample.get("question", "")),
        "prompt": prompt,
        "prompt_type": prompt_type,
        "api_model": model,
        "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
        "raw_model_output": raw_output,
        "raw_extracted_actions": raw_actions,
        "generated": actions,
        "model_output": actions,
    }
    if len(action_parts) > action_limit:
        row["postprocess_truncated_actions"] = {
            "raw_action_count": len(action_parts),
            "kept_action_count": action_limit,
        }
    if api_error:
        row["api_error"] = api_error
    if api_metadata:
        row["api_metadata"] = api_metadata
    for key in ("world", "start", "goal", "goals", "obstacles", "shortest_path_length"):
        if key in sample:
            row[key] = sample[key]
    return row


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    """Load completed or partial API results so interrupted runs can resume."""
    if not path.exists():
        return {}
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("id"))] = row
    return rows


async def call_deepseek(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    sample: dict[str, Any],
    prompt: str,
) -> tuple[str, str | None, dict[str, Any]]:
    """Call the DeepSeek chat API with retry and metadata capture.

    Returns raw content, an optional error string, and compact response metadata.
    Client or server errors are retried unless they are clearly non-retryable.
    """
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": args.thinking},
        "temperature": 0,
        "max_tokens": max_tokens_for_sample(args, sample),
        "stream": False,
    }
    if args.prompt_type in DIRECT_PROMPTS:
        payload["stop"] = ["\n"]
    last_error: str | None = None
    for attempt in range(args.retries + 1):
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = str(message.get("content") or "")
            reasoning_content = str(message.get("reasoning_content") or "")
            metadata = {
                "response_id": data.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "reasoning_content_chars": len(reasoning_content),
            }
            if content:
                return content, None, metadata
            if reasoning_content:
                preview = reasoning_content[:240].replace("\n", " ")
                return "", (
                    "empty_content_with_reasoning "
                    f"finish_reason={choice.get('finish_reason')} "
                    f"reasoning_preview={preview!r}"
                ), metadata
            return "", f"empty_content finish_reason={choice.get('finish_reason')}", metadata
        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPStatusError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            last_error = repr(exc)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status not in {408, 409, 429}:
                break
            await asyncio.sleep(min(args.retry_max_sleep, args.retry_base_sleep * (2**attempt)))
    return "", last_error, {}


async def run_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run concurrent API prompting and write resumable partial outputs."""
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"environment variable {args.api_key_env} is not set")

    samples = load_original_samples(args.dataset, args.max_samples)
    demo_samples = load_original_samples(args.demo_dataset, 3)
    prompts = [api_prompt(sample, args.prompt_type, build_prompt(sample, args.prompt_type, demo_samples)) for sample in samples]
    existing = load_existing(args.out) | load_existing(args.partial_out)
    rows: list[dict[str, Any] | None] = [None] * len(samples)
    pending: list[tuple[int, dict[str, Any], str]] = []
    for index, sample in enumerate(samples):
        row_id = str(sample.get("id", sample.get("example_id", index)))
        if row_id in existing:
            rows[index] = existing[row_id]
        else:
            pending.append((index, sample, prompts[index]))

    args.partial_out.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=timeout, limits=limits) as client:
        progress = tqdm(total=len(pending), desc=f"{args.model}/{args.prompt_type}")

        async def worker(index: int, sample: dict[str, Any], prompt: str) -> None:
            async with semaphore:
                raw_output, api_error, api_metadata = await call_deepseek(client, args, sample, prompt)
                row = prediction_row(
                    sample,
                    index,
                    prompt,
                    args.prompt_type,
                    raw_output,
                    args.model,
                    api_error,
                    api_metadata,
                )
                rows[index] = row
                with args.partial_out.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                progress.update(1)

        tasks = [asyncio.create_task(worker(index, sample, prompt)) for index, sample, prompt in pending]
        if tasks:
            await asyncio.gather(*tasks)
        progress.close()

    final_rows = [row for row in rows if row is not None]
    if len(final_rows) != len(samples):
        raise RuntimeError(f"only wrote {len(final_rows)}/{len(samples)} predictions")
    write_jsonl(args.out, final_rows)
    return final_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek API prompting.")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DS_API")
    parser.add_argument("--prompt-type", choices=PROMPT_TYPES, required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--demo-dataset", type=Path, default=Path("data/generated/phase0/jsonl/train.jsonl"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--partial-out", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-base-sleep", type=float, default=1.0)
    parser.add_argument("--retry-max-sleep", type=float, default=30.0)
    args = parser.parse_args()
    if args.partial_out is None:
        args.partial_out = args.out.with_name(args.out.stem + ".partial.jsonl")
    return args


def main() -> None:
    args = parse_args()
    start = time.time()
    rows = asyncio.run(run_async(args))
    print(json.dumps({"wrote": str(args.out), "num_predictions": len(rows), "seconds": time.time() - start}, indent=2))


if __name__ == "__main__":
    main()

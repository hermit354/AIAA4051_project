#!/usr/bin/env python3
"""Build chosen/rejected executor preference pairs for DPO-style training.

Pipeline stage: preference-data construction after a model can generate candidate plans.
Inputs: source records, model checkpoint, tokenizer, and executor scoring settings.
Outputs: JSONL preference rows with chosen and rejected candidate actions.
Typical caller: scripts/run_phase2_executor_feedback_variants.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, normalize_action_text, write_json, write_jsonl
from src.eval.evaluate_predictions import ACTION_DELTAS, execute_plan


class PlanningDataset(Dataset):
    """Tokenized planning dataset used to sample preference candidates."""

    def __init__(self, samples: list[dict[str, Any]], tokenizer: Any, max_source_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        question = sample.get("question", sample.get("nl_description", sample.get("natural_input", "")))
        encoded = self.tokenizer(str(question).strip(), max_length=self.max_source_length, truncation=True)
        encoded["index"] = index
        return encoded


def collate_without_index(collator: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tokenized rows while preserving original dataset indices."""
    indices = [feature.pop("index") for feature in features]
    batch = collator(features)
    batch["index"] = torch.tensor(indices, dtype=torch.long)
    return batch


def select_device() -> torch.device:
    """Use CUDA for candidate generation when available."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def prediction_record(sample: dict[str, Any], candidate: str, index: int) -> dict[str, Any]:
    """Build an evaluator-compatible record for one preference candidate."""
    return {
        **sample,
        "id": sample.get("id", sample.get("example_id", index)),
        "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
        "model_output": normalize_action_text(candidate),
    }


def reward(result: dict[str, Any]) -> float:
    """Assign an executor-based scalar score to a candidate prediction."""
    parsed = result["parsed_actions"]
    shortest = result["shortest_path_length"]
    score = sum(1.0 if token in ACTION_DELTAS else -5.0 for token in parsed)
    score += sum(1.0 for step in result.get("trace", []) if step.get("status") != "OUT_OF_BOUNDS")
    score += sum(1.0 for step in result.get("trace", []) if step.get("status") not in {"OUT_OF_BOUNDS", "HIT_OBSTACLE"})
    if result["success"]:
        score += 10.0
    if result["optimal"]:
        score += 5.0
    if shortest is not None and result["generated_path_length"] > int(shortest):
        score -= float(result["generated_path_length"] - int(shortest))
    return score


def action_text(result: dict[str, Any]) -> str:
    """Extract clean action text from an evaluated candidate result."""
    return " ".join(action for action in result["parsed_actions"] if action in ACTION_DELTAS)


def rank_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int, int, float, int]:
    """Sort candidates by reward, success, optimality, distance, and length."""
    _candidate, result = item
    distance = result["final_distance_to_goal"]
    if distance is None:
        distance = 1_000_000
    return (
        reward(result),
        int(result["success"]),
        int(result["optimal"]),
        -float(distance),
        -int(result["generated_path_length"]),
    )


def preference_row(sample: dict[str, Any], candidates: list[str], index: int) -> dict[str, Any]:
    """Create one chosen/rejected row from executor-ranked candidates."""
    scored = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_action_text(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        result = execute_plan(prediction_record(sample, normalized, index), index)
        scored.append((normalized, result))
    if not scored:
        oracle = normalize_action_text(sample.get("agent_as_a_point", sample.get("target", "")))
        return {**sample, "chosen": oracle, "rejected": "", "chosen_reward": 0.0, "rejected_reward": -100.0}

    best_candidate, best_result = max(scored, key=rank_key)
    worst_candidate, worst_result = min(scored, key=rank_key)
    oracle = normalize_action_text(sample.get("agent_as_a_point", sample.get("target", "")))
    chosen = action_text(best_result)
    chosen_source = "candidate"
    if not best_result["success"] or not chosen:
        chosen = oracle
        chosen_source = "oracle"
    rejected = action_text(worst_result) or normalize_action_text(worst_candidate)
    if rejected == chosen:
        rejected = action_text(best_result) if chosen_source == "oracle" else normalize_action_text(worst_candidate)
    return {
        **sample,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_source": chosen_source,
        "chosen_reward": reward(best_result) if chosen_source == "candidate" else reward(best_result) + 1.0,
        "rejected_reward": reward(worst_result),
        "chosen_failure_type": best_result["failure_type"],
        "rejected_failure_type": worst_result["failure_type"],
        "raw_best_candidate": normalize_action_text(best_candidate),
        "raw_worst_candidate": normalize_action_text(worst_candidate),
    }


@torch.no_grad()
def generate_split(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    """Generate preference rows for one split with batched beam candidates."""
    samples = load_records(args.source_dir / f"{split}.jsonl")
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True).to(device)
    model.eval()
    dataset = PlanningDataset(samples, tokenizer, args.max_source_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_without_index(collator, features),
    )
    rows: list[dict[str, Any] | None] = [None] * len(samples)
    for batch in tqdm(loader, desc=f"prefs/{split}"):
        indices = batch.pop("index").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for offset, index in enumerate(indices):
            rows[index] = preference_row(samples[index], decoded[offset * args.num_beams : (offset + 1) * args.num_beams], index)
    return [row for row in rows if row is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create executor preference pairs.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/phase2_executor_preferences"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase1_baselines_flan_bf16/flan-t5-base/best"))
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {"source_dir": str(args.source_dir), "checkpoint": str(args.checkpoint), "counts": {}}
    for split in args.splits:
        rows = generate_split(args, split)
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        manifest["counts"][split] = len(rows)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

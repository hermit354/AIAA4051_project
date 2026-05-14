#!/usr/bin/env python3
"""Build executor-feedback SFT data by reranking model candidates.

Pipeline stage: dataset construction that uses a trained model and the executor.
Inputs: source grid records, a seq2seq checkpoint, and candidate-generation settings.
Outputs: JSONL examples where the target reflects executor-preferred actions.
Typical caller: scripts/run_phase2_executor_feedback_sft.sh.
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
from src.eval.evaluate_predictions import ACTION_DELTAS, execute_plan, tokenize_actions


DEFAULT_TEST_SPLITS = [
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


class PlanningDataset(Dataset):
    """Tokenized planning dataset used to generate candidate plans."""

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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prediction_record(sample: dict[str, Any], candidate: str, index: int) -> dict[str, Any]:
    """Build an evaluator-compatible record for one generated candidate."""
    return {
        **sample,
        "id": sample.get("id", sample.get("example_id", index)),
        "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
        "model_output": normalize_action_text(candidate),
    }


def executor_reward(result: dict[str, Any]) -> float:
    """Assign a scalar reward from executor behavior.

    The reward favors legal actions, successful completion, optimality, and
    shorter paths while penalizing invalid tokens, long detours, and loops.
    """
    parsed_actions = result["parsed_actions"]
    shortest = result["shortest_path_length"]
    reward = 0.0
    for token in parsed_actions:
        if token in ACTION_DELTAS:
            reward += 1.0
        else:
            reward -= 5.0
    trace = result.get("trace", [])
    reward += sum(1.0 for step in trace if step.get("status") != "OUT_OF_BOUNDS")
    reward += sum(1.0 for step in trace if step.get("status") not in {"OUT_OF_BOUNDS", "HIT_OBSTACLE"})
    if result["success"]:
        reward += 10.0
    if result["optimal"]:
        reward += 5.0
    if shortest is not None and result["generated_path_length"] > int(shortest):
        reward -= float(result["generated_path_length"] - int(shortest))
    if result["failure_type"] == "LOOP_TOO_LONG":
        reward -= 5.0
    return reward


def candidate_key(result: dict[str, Any]) -> tuple[float, int, float, int]:
    """Sort key used to select the best executor-scored candidate."""
    distance = result["final_distance_to_goal"]
    if distance is None:
        distance = 1_000_000
    return (
        executor_reward(result),
        int(result["success"]),
        -float(distance),
        -int(result["generated_path_length"]),
    )


def clean_actions_for_target(result: dict[str, Any]) -> str:
    """Convert an evaluated candidate back into a clean action target."""
    return " ".join(action for action in result["parsed_actions"] if action in ACTION_DELTAS)


def select_target(sample: dict[str, Any], candidates: list[str], index: int) -> dict[str, Any]:
    """Select the SFT target from generated candidates or fall back to oracle."""
    scored = []
    for candidate in candidates:
        result = execute_plan(prediction_record(sample, candidate, index), index)
        scored.append((candidate, result))
    best_candidate, best_result = max(scored, key=lambda item: candidate_key(item[1]))
    oracle = normalize_action_text(sample.get("agent_as_a_point", sample.get("target", "")))

    selected = clean_actions_for_target(best_result)
    selected_source = "model_candidate"
    if not best_result["success"] or not selected:
        selected = oracle
        selected_source = "oracle_fallback"

    return {
        "selected_target": selected,
        "selected_source": selected_source,
        "selected_reward": executor_reward(best_result),
        "selected_failure_type": best_result["failure_type"],
        "selected_success": bool(best_result["success"]),
        "selected_optimal": bool(best_result["optimal"]),
        "raw_selected_candidate": normalize_action_text(best_candidate),
    }


@torch.no_grad()
def generate_feedback_split(
    records: list[dict[str, Any]],
    checkpoint: Path,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate one executor-feedback split with batched model candidates."""
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True).to(device)
    model.eval()

    dataset = PlanningDataset(records, tokenizer, max_source_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_without_index(collator, features),
    )

    converted: list[dict[str, Any] | None] = [None] * len(records)
    counts = {"model_candidate": 0, "oracle_fallback": 0}
    for batch in tqdm(dataloader, desc="executor-feedback"):
        indices = batch.pop("index").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for offset, index in enumerate(indices):
            candidates = decoded[offset * num_beams : (offset + 1) * num_beams]
            feedback = select_target(records[index], candidates, index)
            row = dict(records[index])
            row["sft_method"] = "executor_feedback"
            row["executor_feedback"] = feedback
            row["target"] = feedback["selected_target"]
            counts[feedback["selected_source"]] += 1
            converted[index] = row

    return [row for row in converted if row is not None], counts


def copy_eval_split(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy non-feedback evaluation splits while tagging the SFT method."""
    copied = []
    for record in records:
        row = dict(record)
        row["sft_method"] = "executor_feedback"
        copied.append(row)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create executor-feedback SFT data.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/generated/phase0/jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/phase2_sft_method/executor_feedback"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase1_baselines_flan_bf16/flan-t5-base/best"))
    parser.add_argument("--feedback-splits", nargs="*", default=["train", "val"])
    parser.add_argument("--test-splits", nargs="*", default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {
        "source_dir": str(args.source_dir),
        "checkpoint": str(args.checkpoint),
        "num_beams": args.num_beams,
        "counts": {},
        "feedback_counts": {},
    }
    jsonl_root = args.output_dir / "jsonl"
    for split in args.feedback_splits:
        records = load_records(args.source_dir / f"{split}.jsonl")
        converted, feedback_counts = generate_feedback_split(
            records,
            args.checkpoint,
            args.batch_size,
            args.max_source_length,
            args.max_new_tokens,
            args.num_beams,
        )
        write_jsonl(jsonl_root / f"{split}.jsonl", converted)
        manifest["counts"][split] = len(converted)
        manifest["feedback_counts"][split] = feedback_counts

    for split in args.test_splits:
        records = copy_eval_split(load_records(args.source_dir / f"{split}.jsonl"))
        write_jsonl(jsonl_root / f"{split}.jsonl", records)
        manifest["counts"][split] = len(records)

    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

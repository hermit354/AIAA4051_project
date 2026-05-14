#!/usr/bin/env python3
"""Run decoding-strategy ablations for seq2seq grid planners.

Pipeline stage: inference after a seq2seq checkpoint has been trained.
Inputs: checkpoint directory, dataset JSON/JSONL, and generation settings.
Outputs: prediction JSONL files for later executor evaluation.
Typical caller: scripts/run_decoding_ablation.sh.
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

from src.data.ppnl_io import load_original_samples, normalize_action_text, write_jsonl
from src.eval.evaluate_predictions import ACTION_DELTAS, execute_plan


ACTION_ORDER = ["up", "down", "left", "right"]


class InferenceDataset(Dataset):
    """Tokenized seq2seq inference dataset with original row indices."""

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


def select_device() -> torch.device:
    """Prefer CUDA for decoding ablations and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_without_index(collator: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tokenized examples while preserving dataset row indices."""
    indices = [feature.pop("index") for feature in features]
    batch = collator(features)
    batch["index"] = torch.tensor(indices, dtype=torch.long)
    return batch


def single_token_action_ids(tokenizer: Any) -> dict[int, str]:
    """Map tokenizer ids to actions for constrained decoding.

    Token-level constraints only work when each action word is represented by a
    single token for the selected tokenizer.
    """
    mapping = {}
    for action in ACTION_ORDER:
        ids = tokenizer.encode(action, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"action {action!r} is not single-token for {type(tokenizer).__name__}: {ids}")
        mapping[int(ids[0])] = action
    return mapping


def clean_action_text(text: str) -> str:
    """Keep only supported action tokens from generated text."""
    tokens = normalize_action_text(text.replace(",", " ")).split()
    return " ".join(token for token in tokens if token in ACTION_DELTAS)


def prediction_row(sample: dict[str, Any], index: int, generated: str, strategy: str) -> dict[str, Any]:
    """Build one evaluator-compatible prediction row."""
    row = {
        "id": sample.get("id", sample.get("example_id", index)),
        "english": sample.get("nl_description", sample.get("question", sample.get("natural_input", ""))),
        "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
        "generated": normalize_action_text(generated),
        "model_output": normalize_action_text(generated),
        "decoding_strategy": strategy,
    }
    for key in ("world", "start", "goal", "goals", "obstacles", "shortest_path_length"):
        if key in sample:
            row[key] = sample[key]
    return row


def score_candidate(sample: dict[str, Any], candidate: str, index: int) -> tuple[int, float, int]:
    """Score a generated candidate by executor behavior for reranking."""
    result = execute_plan(prediction_row(sample, index, candidate, "candidate"), index)
    if result["failure_type"] == "SUCCESS_OPTIMAL":
        rank = 0
    elif result["success"]:
        rank = 1
    elif result["feasible"]:
        rank = 2
    else:
        rank = 3
    distance = result["final_distance_to_goal"]
    if distance is None:
        distance = 1_000_000
    return rank, float(distance), result["generated_path_length"]


def state_from_prefix(sample: dict[str, Any], token_ids: list[int], action_by_id: dict[int, str]) -> tuple[int, int]:
    """Simulate the valid prefix actions to recover the current grid state."""
    current = tuple(sample["start"])
    obstacles = {tuple(item) for item in sample.get("obstacles", [])}
    rows, cols = sample["grid_size"] if "grid_size" in sample else (len(sample["world"]), len(sample["world"][0]))
    for token_id in token_ids:
        action = action_by_id.get(int(token_id))
        if action is None:
            continue
        dr, dc = ACTION_DELTAS[action]
        nxt = (current[0] + dr, current[1] + dc)
        if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or nxt in obstacles:
            break
        current = nxt
    return current


def allowed_state_actions(sample: dict[str, Any], token_ids: list[int], action_by_id: dict[int, str], eos_id: int) -> list[int]:
    """Return token ids that are valid from the state implied by the prefix."""
    current = state_from_prefix(sample, token_ids, action_by_id)
    goal = tuple(sample.get("goal", sample.get("goals", [[None, None]])[0]))
    if current == goal:
        return [eos_id]
    obstacles = {tuple(item) for item in sample.get("obstacles", [])}
    rows, cols = sample["grid_size"] if "grid_size" in sample else (len(sample["world"]), len(sample["world"][0]))
    allowed = []
    for token_id, action in action_by_id.items():
        dr, dc = ACTION_DELTAS[action]
        nxt = (current[0] + dr, current[1] + dc)
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in obstacles:
            allowed.append(token_id)
    return allowed or [eos_id]


@torch.no_grad()
def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the selected decoding strategy and return prediction rows."""
    device = select_device()
    samples = load_original_samples(args.dataset, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True).to(device)
    model.eval()

    dataset = InferenceDataset(samples, tokenizer, args.max_source_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_without_index(collator, features),
    )
    action_by_id = single_token_action_ids(tokenizer)
    action_token_ids = sorted(action_by_id)
    eos_id = int(tokenizer.eos_token_id)
    predictions: list[dict[str, Any] | None] = [None] * len(samples)

    for batch in tqdm(dataloader, desc=args.strategy):
        indices = batch.pop("index").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}

        generate_kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            "max_new_tokens": args.max_new_tokens,
        }

        if args.strategy == "beam4":
            generate_kwargs.update({"num_beams": 4})
        elif args.strategy == "beam8":
            generate_kwargs.update({"num_beams": 8})
        elif args.strategy == "token_constrained":
            allowed = action_token_ids + [eos_id]
            generate_kwargs["prefix_allowed_tokens_fn"] = lambda _batch_id, _sent: allowed
        elif args.strategy == "state_aware":
            batch_samples = [samples[index] for index in indices]

            def prefix_allowed(batch_id: int, sent: torch.Tensor) -> list[int]:
                token_ids = [int(item) for item in sent.tolist()]
                return allowed_state_actions(batch_samples[batch_id], token_ids, action_by_id, eos_id)

            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed
        elif args.strategy == "rerank_beam8":
            generate_kwargs.update({"num_beams": 8, "num_return_sequences": 8})

        generated_ids = model.generate(**generate_kwargs)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if args.strategy == "rerank_beam8":
            for offset, index in enumerate(indices):
                candidates = decoded[offset * 8 : (offset + 1) * 8]
                best = min(candidates, key=lambda text: score_candidate(samples[index], text, index))
                predictions[index] = prediction_row(samples[index], index, best, args.strategy)
        else:
            for index, generated in zip(indices, decoded):
                if args.strategy == "postprocess":
                    generated = clean_action_text(generated)
                predictions[index] = prediction_row(samples[index], index, generated, args.strategy)

    return [item for item in predictions if item is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seq2seq decoding ablations.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=["greedy", "beam4", "beam8", "token_constrained", "postprocess", "rerank_beam8", "state_aware"],
        default="greedy",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = run(args)
    write_jsonl(args.out, predictions)
    print(json.dumps({"wrote": str(args.out), "num_predictions": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()

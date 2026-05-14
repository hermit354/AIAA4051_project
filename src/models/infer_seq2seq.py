#!/usr/bin/env python3
"""Generate full action-sequence predictions from a seq2seq checkpoint.

Pipeline stage: direct inference after supervised training.
Inputs: checkpoint directory, dataset JSON/JSONL, and generation settings.
Outputs: prediction JSONL or JSON files containing model_output fields.
Typical caller: run_phase1_baselines.py and Phase 2 runner scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

from src.data.ppnl_io import load_original_samples, normalize_action_text, write_json, write_jsonl

ACTION_MAP = {"up": "up", "down": "down", "left": "left", "right": "right", "u": "up", "d": "down", "l": "left", "r": "right"}
ACTION_PATTERN = re.compile(r"\b(?:up|down|left|right|u|d|l|r)\b", re.IGNORECASE)
FINAL_MARKERS = ("final answer:", "final actions:", "final:", "final ")


def extract_action_words(text: str) -> str:
    lowered = text.lower()
    segment = text
    marker_positions = [(lowered.rfind(marker), marker) for marker in FINAL_MARKERS]
    marker_positions = [(pos, marker) for pos, marker in marker_positions if pos >= 0]
    if marker_positions:
        pos, marker = max(marker_positions)
        segment = text[pos + len(marker) :]
    return " ".join(ACTION_MAP[token.lower()] for token in ACTION_PATTERN.findall(segment))


class InferenceDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], tokenizer: Any, max_source_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        question = sample.get("question", sample.get("nl_description", sample.get("english", "")))
        encoded = self.tokenizer(
            str(question).strip(),
            max_length=self.max_source_length,
            truncation=True,
        )
        encoded["index"] = index
        return encoded


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_without_index(collator: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    indices = [feature.pop("index") for feature in features]
    batch = collator(features)
    batch["index"] = torch.tensor(indices, dtype=torch.long)
    return batch


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = select_device()
    print(f"device={device}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')}")

    samples = load_original_samples(args.dataset, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True)
    model.to(device)
    model.eval()

    dataset = InferenceDataset(samples, tokenizer, args.max_source_length)
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_without_index(base_collator, features),
        num_workers=args.num_workers,
    )

    predictions: list[dict[str, Any] | None] = [None] * len(samples)
    for batch in tqdm(dataloader, desc="infer"):
        indices = batch.pop("index").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for index, generated in zip(indices, decoded):
            sample = samples[index]
            model_output = extract_action_words(generated) if args.extract_actions else normalize_action_text(generated)
            row = {
                "id": sample.get("id", index),
                "english": sample.get("nl_description", sample.get("question", "")),
                "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
                "raw_generated": normalize_action_text(generated),
                "generated": model_output,
                "model_output": model_output,
            }
            for key in ("world", "start", "goals", "obstacles"):
                if key in sample:
                    row[key] = sample[key]
            predictions[index] = row

    return [item for item in predictions if item is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seq2seq inference on a PPNL sample file.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--extract-actions", action="store_true")
    args = parser.parse_args()

    predictions = run_inference(args)
    if args.out.suffix.lower() == ".jsonl":
        write_jsonl(args.out, predictions)
    else:
        write_json(args.out, predictions)
    print(json.dumps({"wrote": str(args.out), "num_predictions": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()

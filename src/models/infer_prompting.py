#!/usr/bin/env python3
"""Run prompt-only inference on local language models.

Pipeline stage: non-finetuned local-model baseline inference.
Inputs: local model name/path, dataset records, prompt style, and generation settings.
Outputs: prediction JSONL files for executor evaluation.
Typical caller: scripts/run_phase2_prompting.sh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_original_samples, normalize_action_text, write_jsonl
from src.models.model_registry import model_path


ACTION_PATTERN = re.compile(r"\b(?:up|down|left|right)\b", re.IGNORECASE)
FINAL_MARKERS = (
    "final answer:",
    "final actions:",
    "answer:",
    "actions:",
    "path:",
)


class PromptDataset(Dataset):
    """Tokenized prompt dataset that preserves original sample indices."""

    def __init__(self, samples: list[dict[str, Any]], prompts: list[str], tokenizer: Any, max_source_length: int) -> None:
        self.samples = samples
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(self.prompts[index], max_length=self.max_source_length, truncation=True)
        encoded["index"] = index
        return encoded


def collate_without_index(collator: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tokenized prompts while keeping row indices for output alignment."""
    indices = [feature.pop("index") for feature in features]
    batch = collator(features)
    batch["index"] = torch.tensor(indices, dtype=torch.long)
    return batch


def prompt_body(sample: dict[str, Any], structured: bool = False) -> str:
    """Select the text representation used inside a prompt."""
    if structured:
        return str(sample.get("structured_input", sample.get("question", sample.get("nl_description", "")))).strip()
    return str(sample.get("question", sample.get("nl_description", sample.get("natural_input", "")))).strip()


def demonstration(sample: dict[str, Any], structured: bool = False) -> str:
    """Format one labeled example for one-shot or few-shot prompting."""
    return (
        f"Input: {prompt_body(sample, structured=structured)}\n"
        f"Answer: {normalize_action_text(sample.get('target', sample.get('agent_as_a_point', '')))}"
    )


def build_prompt(sample: dict[str, Any], prompt_type: str, demos: list[dict[str, Any]]) -> str:
    """Construct the prompt template for one path-planning example.

    Direct prompt types ask for only the action sequence. CoT and ReAct prompt
    types allow intermediate text but require a final answer marker so
    `extract_actions` can recover the executable action string.
    """
    instruction = (
        "You are solving a grid path-planning task. Return a shortest valid path using only these action tokens: "
        "up, down, left, right."
    )
    output_rule = "Return only the action sequence separated by spaces."
    if prompt_type == "zero_shot":
        return f"{instruction}\n{output_rule}\nInput: {prompt_body(sample)}\nAnswer:"
    if prompt_type == "one_shot":
        return f"{instruction}\n{output_rule}\n\n{demonstration(demos[0])}\n\nInput: {prompt_body(sample)}\nAnswer:"
    if prompt_type == "few_shot":
        demo_text = "\n\n".join(demonstration(demo) for demo in demos[:3])
        return f"{instruction}\n{output_rule}\n\n{demo_text}\n\nInput: {prompt_body(sample)}\nAnswer:"
    if prompt_type == "cot":
        return (
            f"{instruction}\nThink step by step about the route, then end with a line exactly like "
            f"'Final answer: <actions>'.\nInput: {prompt_body(sample)}\nReasoning:"
        )
    if prompt_type == "react":
        return (
            f"{instruction}\nUse repeated Thought/Action notes if useful, then end with a line exactly like "
            f"'Final answer: <actions>'.\nInput: {prompt_body(sample)}\nThought:"
        )
    if prompt_type == "structured":
        return f"{instruction}\n{output_rule}\nInput: {prompt_body(sample, structured=True)}\nAnswer:"
    raise ValueError(f"unknown prompt type: {prompt_type}")


def extract_actions(text: str) -> str:
    """Extract executable path actions from raw model text."""
    lowered = text.lower()
    segment = text
    marker_positions = [(lowered.rfind(marker), marker) for marker in FINAL_MARKERS]
    marker_positions = [(pos, marker) for pos, marker in marker_positions if pos >= 0]
    if marker_positions:
        pos, marker = max(marker_positions)
        segment = text[pos + len(marker) :]
    actions = ACTION_PATTERN.findall(segment)
    return " ".join(action.lower() for action in actions)


def is_encoder_decoder(model_dir: Path) -> bool:
    """Return whether a local Hugging Face model uses encoder-decoder generation."""
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    return bool(getattr(config, "is_encoder_decoder", False))


def dtype_for_device(device: torch.device) -> torch.dtype | None:
    """Use bfloat16 on CUDA for memory efficiency; keep CPU inference in fp32."""
    if device.type != "cuda":
        return None
    return torch.bfloat16


def select_device() -> torch.device:
    """Prefer CUDA for local prompting and fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run batched local prompting and return evaluator-compatible rows."""
    samples = load_original_samples(args.dataset, args.max_samples)
    demo_samples = load_original_samples(args.demo_dataset, 3)
    prompts = [build_prompt(sample, args.prompt_type, demo_samples) for sample in samples]

    model_dir = args.model_path or model_path(args.model)
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder_decoder = is_encoder_decoder(model_dir)
    dtype = dtype_for_device(device)
    model_cls = AutoModelForSeq2SeqLM if encoder_decoder else AutoModelForCausalLM
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = model_cls.from_pretrained(model_dir, **model_kwargs).to(device)
    model.eval()

    dataset = PromptDataset(samples, prompts, tokenizer, args.max_source_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_without_index(collator, features),
    )

    predictions: list[dict[str, Any] | None] = [None] * len(samples)
    for batch in tqdm(dataloader, desc=f"{args.model}/{args.prompt_type}"):
        indices = batch.pop("index").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if encoder_decoder:
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            decoded = []
            for row_ids, input_ids in zip(generated_ids, batch["input_ids"]):
                continuation = row_ids[len(input_ids) :]
                decoded.append(tokenizer.decode(continuation, skip_special_tokens=True))
        for index, raw_output in zip(indices, decoded):
            sample = samples[index]
            actions = extract_actions(raw_output)
            row = {
                "id": sample.get("id", sample.get("example_id", index)),
                "english": sample.get("nl_description", sample.get("question", "")),
                "prompt": prompts[index],
                "prompt_type": args.prompt_type,
                "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
                "raw_model_output": raw_output,
                "generated": actions,
                "model_output": actions,
            }
            for key in ("world", "start", "goal", "goals", "obstacles", "shortest_path_length"):
                if key in sample:
                    row[key] = sample[key]
            predictions[index] = row
    return [row for row in predictions if row is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-only local model inference.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--prompt-type", choices=["zero_shot", "one_shot", "few_shot", "cot", "react", "structured"], required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--demo-dataset", type=Path, default=Path("data/generated/phase0/jsonl/train.jsonl"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    write_jsonl(args.out, rows)
    print(json.dumps({"wrote": str(args.out), "num_predictions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

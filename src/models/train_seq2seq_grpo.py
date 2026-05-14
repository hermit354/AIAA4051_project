#!/usr/bin/env python3
"""Train a seq2seq grid planner with GRPO/RLVR-style executor rewards.

Pipeline stage: reward-based training variant after supervised baselines exist.
Inputs: JSON config files, JSONL training data, and executor reward settings.
Outputs: reward-trained checkpoints, metrics, and training summaries.
Typical caller: scripts/run_phase2_executor_feedback_variants.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, get_scheduler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_jsonl, normalize_action_text, write_json
from src.eval.evaluate_predictions import ACTION_DELTAS, execute_plan


@dataclass
class GRPOConfig:
    model_name_or_path: str
    reference_model_name_or_path: str
    output_dir: str
    train_path: str
    eval_path: str
    max_source_length: int = 160
    max_new_tokens: int = 96
    group_size: int = 4
    num_train_epochs: int = 3
    max_steps: int | None = 3000
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.03
    kl_coef: float = 0.02
    eval_steps: int = 250
    early_stopping_patience: int = 8
    seed: int = 1
    bf16: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = 512


class PlanningDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_source_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        question = row.get("question", row.get("nl_description", row.get("natural_input", "")))
        encoded = self.tokenizer(str(question).strip(), max_length=self.max_source_length, truncation=True)
        encoded["index"] = index
        return encoded


def load_config(path: Path) -> GRPOConfig:
    raw = json.load(path.open())
    valid = set(GRPOConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {unknown}")
    return GRPOConfig(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def collate_without_index(tokenizer: Any, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    indices = [feature.pop("index") for feature in features]
    batch = DataCollatorWithPadding(tokenizer=tokenizer)(features)
    batch["index"] = torch.tensor(indices, dtype=torch.long)
    return batch


def prediction_record(sample: dict[str, Any], candidate: str, index: int) -> dict[str, Any]:
    return {
        **sample,
        "id": sample.get("id", sample.get("example_id", index)),
        "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
        "model_output": normalize_action_text(candidate),
    }


def executor_reward(result: dict[str, Any]) -> float:
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


def sequence_log_probs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def encode_responses(tokenizer: Any, responses: list[str], device: torch.device) -> torch.Tensor:
    encoded = tokenizer(responses, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels = encoded["input_ids"].to(device)
    return labels.masked_fill(labels.eq(tokenizer.pad_token_id), -100)


@torch.no_grad()
def generate_groups(model: Any, tokenizer: Any, batch: dict[str, torch.Tensor], cfg: GRPOConfig) -> tuple[list[str], torch.Tensor]:
    generated_ids = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=cfg.group_size,
    )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded, generated_ids


def expand_inputs(batch: dict[str, torch.Tensor], group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].repeat_interleave(group_size, dim=0)
    attention_mask = batch["attention_mask"].repeat_interleave(group_size, dim=0)
    return input_ids, attention_mask


def compute_rewards(samples: list[dict[str, Any]], indices: list[int], responses: list[str], group_size: int) -> torch.Tensor:
    rewards = []
    for offset, index in enumerate(indices):
        sample = samples[index]
        for response in responses[offset * group_size : (offset + 1) * group_size]:
            result = execute_plan(prediction_record(sample, response, index), index)
            rewards.append(executor_reward(result))
    return torch.tensor(rewards, dtype=torch.float32)


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((grouped - mean) / std).view(-1)


@torch.no_grad()
def evaluate_success(model: Any, tokenizer: Any, rows: list[dict[str, Any]], cfg: GRPOConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    dataset = PlanningDataset(rows, tokenizer, cfg.max_source_length)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_without_index(tokenizer, features),
    )
    rewards = []
    successes = []
    optimal = []
    for batch in tqdm(loader, desc="eval", leave=False):
        indices = batch.pop("index").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=cfg.max_new_tokens,
            num_beams=1,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for index, response in zip(indices, decoded):
            result = execute_plan(prediction_record(rows[index], response, index), index)
            rewards.append(executor_reward(result))
            successes.append(float(result["success"]))
            optimal.append(float(result["optimal"]))
    return {
        "reward": sum(rewards) / max(1, len(rewards)),
        "success_rate": sum(successes) / max(1, len(successes)),
        "optimality": sum(optimal) / max(1, len(optimal)),
    }


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, name: str, metrics: dict[str, Any]) -> None:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    write_json(checkpoint_dir / "metrics.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train seq2seq with GRPO/RLVR-style executor reward.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", cfg.__dict__)

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, local_files_only=True)
    policy = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path, local_files_only=True).to(device)
    reference = AutoModelForSeq2SeqLM.from_pretrained(cfg.reference_model_name_or_path, local_files_only=True).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    train_rows = load_jsonl(cfg.train_path)
    eval_rows = load_jsonl(cfg.eval_path)
    if cfg.max_train_samples is not None:
        train_rows = train_rows[: cfg.max_train_samples]
    if cfg.max_eval_samples is not None:
        eval_rows = eval_rows[: cfg.max_eval_samples]
    train_dataset = PlanningDataset(train_rows, tokenizer, cfg.max_source_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda features: collate_without_index(tokenizer, features),
    )
    total_steps = cfg.max_steps or math.ceil(len(train_loader)) * cfg.num_train_epochs
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps)
    history_path = output_dir / "training_history.jsonl"
    best_reward = -float("inf")
    best_step = 0
    no_improve = 0
    global_step = 0

    for epoch in range(cfg.num_train_epochs):
        policy.train()
        progress = tqdm(train_loader, desc=f"grpo epoch {epoch + 1}/{cfg.num_train_epochs}")
        for batch in progress:
            indices = batch.pop("index").tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                responses, _generated_ids = generate_groups(policy, tokenizer, batch, cfg)
            rewards = compute_rewards(train_rows, indices, responses, cfg.group_size).to(device)
            advantages = group_advantages(rewards.detach().cpu(), cfg.group_size).to(device)
            labels = encode_responses(tokenizer, responses, device)
            input_ids, attention_mask = expand_inputs(batch, cfg.group_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
                policy_logp = sequence_log_probs(policy, input_ids, attention_mask, labels)
                with torch.no_grad():
                    ref_logp = sequence_log_probs(reference, input_ids, attention_mask, labels)
                pg_loss = -(advantages * policy_logp).mean()
                kl_penalty = (policy_logp - ref_logp).mean()
                loss = pg_loss + cfg.kl_coef * kl_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", reward=f"{float(rewards.mean().cpu()):.2f}")
            if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                metrics = evaluate_success(policy, tokenizer, eval_rows, cfg, device)
                metrics.update({"step": global_step, "epoch": epoch + 1})
                with history_path.open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                write_json(output_dir / "last_eval_metrics.json", metrics)
                if metrics["reward"] > best_reward:
                    best_reward = metrics["reward"]
                    best_step = global_step
                    no_improve = 0
                    save_checkpoint(policy, tokenizer, output_dir, "best", metrics)
                else:
                    no_improve += 1
                policy.train()
                if no_improve >= cfg.early_stopping_patience:
                    break
            if global_step >= total_steps:
                break
        if global_step >= total_steps or no_improve >= cfg.early_stopping_patience:
            break

    final_metrics = evaluate_success(policy, tokenizer, eval_rows, cfg, device)
    final_metrics.update({"step": global_step, "best_reward": best_reward, "best_step": best_step})
    save_checkpoint(policy, tokenizer, output_dir, "final", final_metrics)
    if final_metrics["reward"] > best_reward:
        save_checkpoint(policy, tokenizer, output_dir, "best", final_metrics)
        best_reward = final_metrics["reward"]
        best_step = global_step
    final_metrics.update({"best_reward": best_reward, "best_step": best_step})
    write_json(output_dir / "train_summary.json", final_metrics)
    print(json.dumps({"final": final_metrics}, indent=2))


if __name__ == "__main__":
    main()

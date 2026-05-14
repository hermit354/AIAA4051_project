#!/usr/bin/env python3
"""Train a seq2seq grid planner with DPO preference optimization.

Pipeline stage: preference-based training variant after executor preference data is built.
Inputs: JSON config files and JSONL chosen/rejected preference pairs.
Outputs: DPO checkpoints, evaluation metrics, and training summaries.
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

from src.data.ppnl_io import load_jsonl, write_json


@dataclass
class DPOConfig:
    model_name_or_path: str
    reference_model_name_or_path: str
    output_dir: str
    train_path: str
    eval_path: str
    beta: float = 0.1
    max_source_length: int = 160
    max_target_length: int = 96
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    eval_steps: int = 500
    save_steps: int = 500
    early_stopping_patience: int = 5
    seed: int = 1
    bf16: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, cfg: DPOConfig) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def encode_target(self, text: str) -> list[int]:
        return self.tokenizer(text, max_length=self.cfg.max_target_length, truncation=True)["input_ids"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        question = row.get("question", row.get("nl_description", row.get("natural_input", "")))
        source = self.tokenizer(str(question).strip(), max_length=self.cfg.max_source_length, truncation=True)
        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "chosen_labels": self.encode_target(str(row["chosen"])),
            "rejected_labels": self.encode_target(str(row["rejected"])),
        }


def load_config(path: Path) -> DPOConfig:
    raw = json.load(path.open())
    valid = set(DPOConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {unknown}")
    return DPOConfig(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def collate_preferences(tokenizer: Any, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    source = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
    source_batch = DataCollatorWithPadding(tokenizer=tokenizer)(source)

    def pad_labels(key: str) -> torch.Tensor:
        values = [torch.tensor(f[key], dtype=torch.long) for f in features]
        padded = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=tokenizer.pad_token_id)
        return padded

    source_batch["chosen_labels"] = pad_labels("chosen_labels")
    source_batch["rejected_labels"] = pad_labels("rejected_labels")
    return source_batch


def sequence_log_probs(model: Any, batch: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=decoder_input_ids,
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def prepare_labels(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    return labels.masked_fill(labels.eq(pad_id), -100)


def dpo_batch_loss(policy: Any, reference: Any, batch: dict[str, torch.Tensor], beta: float, pad_id: int) -> tuple[torch.Tensor, dict[str, float]]:
    chosen = prepare_labels(batch["chosen_labels"], pad_id)
    rejected = prepare_labels(batch["rejected_labels"], pad_id)
    policy_chosen = sequence_log_probs(policy, batch, chosen)
    policy_rejected = sequence_log_probs(policy, batch, rejected)
    with torch.no_grad():
        ref_chosen = sequence_log_probs(reference, batch, chosen)
        ref_rejected = sequence_log_probs(reference, batch, rejected)
    policy_margin = policy_chosen - policy_rejected
    ref_margin = ref_chosen - ref_rejected
    logits = beta * (policy_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "preference_accuracy": float((logits.detach() > 0).float().mean().cpu()),
        "policy_margin": float(policy_margin.detach().mean().cpu()),
        "reference_margin": float(ref_margin.detach().mean().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(policy: Any, reference: Any, loader: DataLoader, beta: float, pad_id: int, device: torch.device) -> dict[str, float]:
    policy.eval()
    losses = []
    accuracies = []
    margins = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        loss, metrics = dpo_batch_loss(policy, reference, batch, beta, pad_id)
        losses.append(float(loss.cpu()))
        accuracies.append(metrics["preference_accuracy"])
        margins.append(metrics["policy_margin"])
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "preference_accuracy": sum(accuracies) / max(1, len(accuracies)),
        "policy_margin": sum(margins) / max(1, len(margins)),
    }


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, name: str, metrics: dict[str, Any]) -> None:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    write_json(checkpoint_dir / "metrics.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train seq2seq with DPO preferences.")
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

    train_rows = load_jsonl(cfg.train_path)[: cfg.max_train_samples]
    eval_rows = load_jsonl(cfg.eval_path)[: cfg.max_eval_samples]
    train_dataset = PreferenceDataset(train_rows, tokenizer, cfg)
    eval_dataset = PreferenceDataset(eval_rows, tokenizer, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda features: collate_preferences(tokenizer, features),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_preferences(tokenizer, features),
    )
    total_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps) * cfg.num_train_epochs
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    history_path = output_dir / "training_history.jsonl"
    best_loss = float("inf")
    best_step = 0
    no_improve = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_train_epochs):
        policy.train()
        progress = tqdm(train_loader, desc=f"dpo epoch {epoch + 1}/{cfg.num_train_epochs}")
        for batch_index, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
                loss, metrics = dpo_batch_loss(policy, reference, batch, cfg.beta, tokenizer.pad_token_id)
                loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            should_step = (batch_index + 1) % cfg.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=f"{metrics['loss']:.4f}", pref_acc=f"{metrics['preference_accuracy']:.3f}")
            if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                eval_metrics = evaluate(policy, reference, eval_loader, cfg.beta, tokenizer.pad_token_id, device)
                eval_metrics.update({"step": global_step, "epoch": epoch + 1})
                with history_path.open("a") as handle:
                    handle.write(json.dumps(eval_metrics) + "\n")
                write_json(output_dir / "last_eval_metrics.json", eval_metrics)
                if eval_metrics["loss"] < best_loss:
                    best_loss = eval_metrics["loss"]
                    best_step = global_step
                    no_improve = 0
                    save_checkpoint(policy, tokenizer, output_dir, "best", eval_metrics)
                else:
                    no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    break
        if no_improve >= cfg.early_stopping_patience:
            break

    final_metrics = evaluate(policy, reference, eval_loader, cfg.beta, tokenizer.pad_token_id, device)
    final_metrics.update({"step": global_step, "best_loss": best_loss, "best_step": best_step})
    save_checkpoint(policy, tokenizer, output_dir, "final", final_metrics)
    if final_metrics["loss"] < best_loss:
        save_checkpoint(policy, tokenizer, output_dir, "best", final_metrics)
        best_loss = final_metrics["loss"]
        best_step = global_step
    final_metrics.update({"best_loss": best_loss, "best_step": best_step})
    write_json(output_dir / "train_summary.json", final_metrics)
    print(json.dumps({"final": final_metrics}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a seq2seq model for single-goal grid path planning.

Pipeline stage: core supervised training for direct or one-step policy models.
Inputs: JSON config files, train/eval JSONL datasets, and a model name or checkpoint path.
Outputs: checkpoints, training summaries, evaluation metrics, and tokenizer/model artifacts.
Typical caller: experiment orchestrators in src/models/run_phase*.py and shell launchers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ppnl_io import load_records, load_seq2seq_examples, normalize_action_text, write_json
from src.eval.evaluate_predictions import execute_plan


ACTION_MAP = {"up": "up", "down": "down", "left": "left", "right": "right", "u": "up", "d": "down", "l": "left", "r": "right"}
ACTION_PATTERN = re.compile(r"\b(?:up|down|left|right|u|d|l|r)\b", re.IGNORECASE)
FINAL_MARKERS = ("final answer:", "final actions:", "final:", "final ")


@dataclass
class TrainConfig:
    """Configuration for one seq2seq training run.

    The config is loaded from JSON so experiment runners can create reproducible
    training jobs without hard-coding hyperparameters inside shell scripts.
    """

    model_name_or_path: str
    output_dir: str
    train_path: str
    eval_path: str
    max_source_length: int = 128
    max_target_length: int = 64
    num_train_epochs: int = 5
    max_steps: int | None = None
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    seed: int = 1
    fp16: bool = False
    bf16: bool = False
    num_beams: int = 1
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    early_stopping_patience: int = 5
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    eval_max_new_tokens: int = 64
    num_workers: int = 0
    selection_metric: str = "exact_match"


def extract_action_words(text: str) -> str:
    lowered = text.lower()
    segment = text
    marker_positions = [(lowered.rfind(marker), marker) for marker in FINAL_MARKERS]
    marker_positions = [(pos, marker) for pos, marker in marker_positions if pos >= 0]
    if marker_positions:
        pos, marker = max(marker_positions)
        segment = text[pos + len(marker) :]
    return " ".join(ACTION_MAP[token.lower()] for token in ACTION_PATTERN.findall(segment))


class TokenizedSeq2SeqDataset(Dataset):
    """Lazy tokenization wrapper for question/target path-planning examples."""

    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, cfg: TrainConfig) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        model_input = self.tokenizer(
            example["question"],
            max_length=self.cfg.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=example["target"],
            max_length=self.cfg.max_target_length,
            truncation=True,
        )
        model_input["labels"] = labels["input_ids"]
        return model_input


def load_config(path: Path) -> TrainConfig:
    """Load and validate a JSON training config."""
    with path.open() as handle:
        raw = json.load(handle)
    valid = set(TrainConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {unknown}")
    return TrainConfig(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, fp16: bool, bf16: bool):
    if bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def make_dataloader(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    cfg: TrainConfig,
    train: bool,
) -> DataLoader:
    """Build a DataLoader with seq2seq padding and optional train shuffling."""
    dataset = TokenizedSeq2SeqDataset(examples, tokenizer, cfg)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    return DataLoader(
        dataset,
        batch_size=cfg.per_device_train_batch_size if train else cfg.per_device_eval_batch_size,
        shuffle=train,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )


@torch.no_grad()
def evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    dataloader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate text and executor metrics on the validation split.

    Text metrics measure direct target matching. Executor metrics run generated
    actions in the grid and better reflect planning behavior.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        total_loss += float(outputs.loss.detach().cpu())
        total_batches += 1

        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=cfg.eval_max_new_tokens,
            num_beams=cfg.num_beams,
        )
        # Decode by batch to avoid one generation call per example during
        # validation, which keeps frequent checkpoint evaluation affordable.
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(normalize_action_text(item) for item in decoded)

    labels = [example["target"] for example in examples[: len(predictions)]]
    exact = sum(int(pred == gold) for pred, gold in zip(predictions, labels)) / max(1, len(labels))
    text_exact = sum(
        int(pred == normalize_action_text(gold)) for pred, gold in zip(predictions, labels)
    ) / max(1, len(labels))
    executor_results = []
    for index, (prediction, record) in enumerate(zip(predictions, records[: len(predictions)])):
        model_output = extract_action_words(prediction) if cfg.selection_metric.startswith("executor_") else prediction
        executor_results.append(execute_plan({**record, "model_output": model_output}, index))

    metrics = {
        "loss": total_loss / max(1, total_batches),
        "exact_match": exact,
        "text_exact_match": text_exact,
        "num_examples": len(labels),
    }
    if executor_results:
        total = len(executor_results)
        metrics.update(
            {
                "executor_success": sum(1 for row in executor_results if row["success"]) / total,
                "executor_feasibility": sum(1 for row in executor_results if row["feasible"]) / total,
                "executor_optimality": sum(1 for row in executor_results if row["optimal"]) / total,
            }
        )
    return metrics


def is_better(metrics: dict[str, Any], best_metrics: dict[str, Any] | None, selection_metric: str) -> bool:
    value = float(metrics[selection_metric])
    if best_metrics is None:
        return True
    best_value = float(best_metrics[selection_metric])
    if selection_metric == "loss":
        return value < best_value
    if value != best_value:
        return value > best_value
    return float(metrics["loss"]) < float(best_metrics["loss"])


def rotate_checkpoints(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-step-*") if path.is_dir()],
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for checkpoint in checkpoints[:-limit]:
        for child in checkpoint.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(checkpoint.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        checkpoint.rmdir()


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, name: str, metrics: dict[str, Any]) -> None:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    write_json(checkpoint_dir / "metrics.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a clean seq2seq PPNL model.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", cfg.__dict__)

    device = select_device()
    print(f"device={device}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')}")

    train_examples = load_seq2seq_examples(cfg.train_path, cfg.max_train_samples)
    eval_examples = load_seq2seq_examples(cfg.eval_path, cfg.max_eval_samples)
    eval_records = load_records(cfg.eval_path)
    if cfg.max_eval_samples is not None:
        eval_records = eval_records[: cfg.max_eval_samples]
    if not train_examples:
        raise ValueError("no training examples loaded")
    if not eval_examples:
        raise ValueError("no evaluation examples loaded")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path, local_files_only=True)
    model.to(device)

    train_loader = make_dataloader(train_examples, tokenizer, model, cfg, train=True)
    eval_loader = make_dataloader(eval_examples, tokenizer, model, cfg, train=False)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.num_train_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and not cfg.bf16 and device.type == "cuda")

    if cfg.selection_metric not in {
        "exact_match",
        "text_exact_match",
        "loss",
        "executor_success",
        "executor_feasibility",
        "executor_optimality",
    }:
        raise ValueError(f"unsupported selection_metric={cfg.selection_metric}")
    best_metrics: dict[str, Any] | None = None
    best_exact = -1.0
    best_step = 0
    no_improve_evals = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    print(
        json.dumps(
            {
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "train_batches_per_epoch": len(train_loader),
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "total_optimizer_steps": total_steps,
                "warmup_steps": warmup_steps,
            },
            indent=2,
        )
    )

    stop_training = False
    for epoch in range(cfg.num_train_epochs):
        if stop_training:
            break
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.num_train_epochs}")
        for batch_index, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, cfg.fp16, cfg.bf16):
                outputs = model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps

            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    f"non-finite training loss at epoch={epoch + 1} batch={batch_index + 1}: {float(loss.detach().cpu())}"
                )

            scaler.scale(loss).backward()

            should_step = (batch_index + 1) % cfg.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach().cpu()) * cfg.gradient_accumulation_steps:.4f}")

            if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                metrics = evaluate(model, tokenizer, eval_examples, eval_records, eval_loader, cfg, device)
                metrics.update({"step": global_step, "epoch": epoch + 1})
                print(json.dumps({"eval": metrics}, indent=2))
                write_json(output_dir / "last_eval_metrics.json", metrics)

                if is_better(metrics, best_metrics, cfg.selection_metric):
                    best_metrics = dict(metrics)
                    best_exact = metrics["exact_match"]
                    best_step = global_step
                    no_improve_evals = 0
                    save_checkpoint(model, tokenizer, output_dir, "best", metrics)
                else:
                    no_improve_evals += 1

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    save_checkpoint(model, tokenizer, output_dir, f"checkpoint-step-{global_step}", metrics)
                    rotate_checkpoints(output_dir, cfg.save_total_limit)

                model.train()
                if no_improve_evals >= cfg.early_stopping_patience:
                    stop_training = True
                    break

            if global_step >= total_steps:
                stop_training = True
                break

    final_metrics = evaluate(model, tokenizer, eval_examples, eval_records, eval_loader, cfg, device)
    final_metrics.update(
        {
            "step": global_step,
            "best_exact_match": best_exact,
            "best_step": best_step,
            "selection_metric": cfg.selection_metric,
            "best_selection_value": None if best_metrics is None else best_metrics[cfg.selection_metric],
        }
    )
    save_checkpoint(model, tokenizer, output_dir, "final", final_metrics)
    if is_better(final_metrics, best_metrics, cfg.selection_metric):
        final_metrics["best_exact_match"] = final_metrics["exact_match"]
        final_metrics["best_step"] = global_step
        final_metrics["best_selection_value"] = float(final_metrics[cfg.selection_metric])
        save_checkpoint(model, tokenizer, output_dir, "best", final_metrics)
    write_json(output_dir / "train_summary.json", final_metrics)
    print(json.dumps({"final": final_metrics}, indent=2))


if __name__ == "__main__":
    main()

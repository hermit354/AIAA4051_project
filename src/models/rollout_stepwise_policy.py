#!/usr/bin/env python3
"""Roll out a one-step policy under simulator control.

Pipeline stage: core stepwise decoding method for size-generalization experiments.
Inputs: a trained one-step seq2seq checkpoint, dataset records, and rollout/search settings.
Outputs: prediction JSONL rows with generated action sequences and rollout metadata.
Typical caller: scripts/run_phase2_stepwise_eval_suite.sh and related stepwise launchers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make_stepwise_policy_dataset import shortest_distances, state_question, target_text
from src.data.ppnl_io import load_original_samples, normalize_action_text, write_jsonl
from src.eval.evaluate_predictions import ACTION_DELTAS


ACTIONS = ["up", "down", "left", "right"]
TARGETS = [*ACTIONS, "stop"]


def ensure_grid_size(sample: dict[str, Any]) -> dict[str, Any]:
    if "grid_size" not in sample:
        world = sample.get("world")
        if isinstance(world, list) and world:
            sample["grid_size"] = [len(world), len(world[0])]
    return sample


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def valid_actions(sample: dict[str, Any], current: tuple[int, int]) -> list[str]:
    """Return actions that keep the agent inside the grid and off obstacles."""
    rows, cols = sample["grid_size"]
    obstacles = {tuple(item) for item in sample.get("obstacles", [])}
    allowed = []
    for action in ACTIONS:
        dr, dc = ACTION_DELTAS[action]
        nxt = (current[0] + dr, current[1] + dc)
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in obstacles:
            allowed.append(action)
    return allowed


def parse_action(text: str) -> str:
    """Extract the first supported action token from generated text."""
    tokens = normalize_action_text(text.replace(",", " ")).split()
    for token in tokens:
        if token in TARGETS:
            return token
    return "stop"


def move(current: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    return current[0] + dr, current[1] + dc


@torch.no_grad()
def greedy_actions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_source_length: int,
) -> list[str]:
    encoded = tokenizer(prompts, max_length=max_source_length, truncation=True, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(**encoded, max_new_tokens=4, num_beams=1)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [parse_action(text) for text in decoded]


def candidate_label(
    action: str,
    current: tuple[int, int],
    distances: dict[tuple[int, int], int] | None,
    target_format: str,
) -> str:
    return target_text(action, current, distances, target_format)


@torch.no_grad()
def score_candidates(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    candidates: list[list[str]],
    current_states: list[tuple[int, int]],
    distance_maps: list[dict[tuple[int, int], int] | None],
    target_format: str,
    device: torch.device,
    max_source_length: int,
) -> list[list[tuple[str, float]]]:
    """Score candidate next actions with normalized target log probability.

    Candidate scoring lets the rollout compare only actions that the executor
    considers relevant. The flattened batch keeps model calls efficient even
    when each example has a different candidate set.
    """
    flat_prompts = []
    flat_targets = []
    flat_actions = []
    owner = []
    for index, (prompt, row_candidates) in enumerate(zip(prompts, candidates)):
        for candidate in row_candidates:
            flat_prompts.append(prompt)
            flat_targets.append(candidate_label(candidate, current_states[index], distance_maps[index], target_format))
            flat_actions.append(candidate)
            owner.append(index)

    if not flat_prompts:
        return [[] for _ in prompts]

    encoded = tokenizer(flat_prompts, max_length=max_source_length, truncation=True, padding=True, return_tensors="pt")
    labels = tokenizer(text_target=flat_targets, padding=True, return_tensors="pt")["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = labels.to(device)

    logits = model(**encoded, labels=labels).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_scores = torch.gather(log_probs, dim=-1, index=labels.masked_fill(labels == -100, 0).unsqueeze(-1)).squeeze(-1)
    mask = labels != -100
    sequence_scores = (token_scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    scored: list[list[tuple[str, float]]] = [[] for _ in prompts]
    for action, index, score in zip(flat_actions, owner, sequence_scores.tolist()):
        scored[index].append((action, float(score)))
    for row in scored:
        row.sort(key=lambda item: item[1], reverse=True)
    return scored


def candidate_set(sample: dict[str, Any], current: tuple[int, int], mode: str) -> list[str]:
    """Build the candidate action set for a rollout mode."""
    if mode == "plain":
        return TARGETS
    allowed = valid_actions(sample, current)
    goal = tuple(sample["goal"])
    if mode == "mask_invalid":
        return [*allowed, "stop"]
    if mode in {"mask_invalid_no_stop", "topk_verifier", "topk_lookahead", "anti_loop"}:
        return ["stop"] if current == goal else allowed
    raise ValueError(f"unknown mode: {mode}")


def distance_after_action(
    current: tuple[int, int],
    action: str,
    distances: dict[tuple[int, int], int] | None,
) -> int:
    if distances is None:
        return 1_000_000
    if action == "stop":
        return distances.get(current, 1_000_000)
    return distances.get(move(current, action), 1_000_000)


def select_scored_action(
    scored: list[tuple[str, float]],
    mode: str,
    current: tuple[int, int],
    distances: dict[tuple[int, int], int] | None,
    trajectory: list[tuple[int, int]],
    top_k: int,
    anti_loop_window: int,
    anti_loop_penalty: float,
) -> str:
    """Choose one action from model-scored candidates under the rollout mode."""
    if not scored:
        return "stop"
    if mode == "topk_verifier":
        shortlist = scored[: max(1, top_k)]
        return min(shortlist, key=lambda item: (distance_after_action(current, item[0], distances), -item[1]))[0]
    if mode == "anti_loop":
        recent = set(trajectory[-max(1, anti_loop_window) :])
        best_action = scored[0][0]
        best_score = -math.inf
        for action, score in scored:
            adjusted = score
            if action != "stop" and move(current, action) in recent:
                adjusted -= anti_loop_penalty
            if adjusted > best_score:
                best_score = adjusted
                best_action = action
        return best_action
    return scored[0][0]


def loop_penalty(
    nxt: tuple[int, int],
    trajectory: list[tuple[int, int]],
    backtrack_penalty: float,
    revisit_penalty: float,
    recent_loop_penalty: float,
    recent_window: int,
) -> float:
    penalty = 0.0
    if len(trajectory) >= 2 and nxt == trajectory[-2]:
        penalty += backtrack_penalty
    penalty += revisit_penalty * sum(1 for cell in trajectory if cell == nxt)
    recent = set(trajectory[-max(1, recent_window) :])
    if nxt in recent:
        penalty += recent_loop_penalty
    return penalty


@torch.no_grad()
def lookahead_action_one(
    sample: dict[str, Any],
    current: tuple[int, int],
    trajectory: list[tuple[int, int]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    distances: dict[tuple[int, int], int] | None,
) -> str:
    """Run bounded single-example lookahead from the current state.

    The method expands short candidate continuations and scores them with a
    mixture of model likelihood, goal distance, and loop penalties. It is used
    when local one-step scores alone are not enough to avoid later failures.
    """
    goal = tuple(sample["goal"])
    if current == goal:
        return "stop"

    prompt = state_question(
        sample,
        [current[0], current[1]],
        include_distance_signals=args.include_distance_signals,
        include_valid_actions=args.include_valid_actions,
        include_history=args.include_history,
        history=trajectory,
        history_window=args.history_window,
        distances=distances if args.include_distance_signals else None,
    )
    candidates = candidate_set(sample, current, "topk_lookahead")
    scored = score_candidates(
        model,
        tokenizer,
        [prompt],
        [candidates],
        [current],
        [distances],
        args.target_format,
        device,
        args.max_source_length,
    )[0][: max(1, args.top_k)]
    if not scored:
        return "stop"

    beams = []
    for action, logprob in scored:
        nxt = move(current, action)
        beams.append(
            {
                "first_action": action,
                "current": nxt,
                "trajectory": [*trajectory, nxt],
                "score": args.lookahead_model_weight * logprob
                - loop_penalty(
                    nxt,
                    trajectory,
                    args.backtrack_penalty,
                    args.revisit_penalty,
                    args.recent_loop_penalty,
                    args.anti_loop_window,
                ),
            }
        )

    for _ in range(max(0, args.lookahead_steps - 1)):
        expanded = [beam for beam in beams if beam["current"] == goal]
        active_beams = [beam for beam in beams if beam["current"] != goal]
        if not active_beams:
            beams = expanded
            break

        prompts = [
            state_question(
                sample,
                [beam["current"][0], beam["current"][1]],
                include_distance_signals=args.include_distance_signals,
                include_valid_actions=args.include_valid_actions,
                include_history=args.include_history,
                history=beam["trajectory"],
                history_window=args.history_window,
                distances=distances if args.include_distance_signals else None,
            )
            for beam in active_beams
        ]
        candidate_rows = [candidate_set(sample, beam["current"], "topk_lookahead") for beam in active_beams]
        scored_rows = score_candidates(
            model,
            tokenizer,
            prompts,
            candidate_rows,
            [beam["current"] for beam in active_beams],
            [distances for _ in active_beams],
            args.target_format,
            device,
            args.max_source_length,
        )
        for beam, scored in zip(active_beams, scored_rows):
            state = beam["current"]
            for action, logprob in scored[: max(1, args.top_k)]:
                nxt = move(state, action)
                expanded.append(
                    {
                        "first_action": beam["first_action"],
                        "current": nxt,
                        "trajectory": [*beam["trajectory"], nxt],
                        "score": beam["score"]
                        + args.lookahead_model_weight * logprob
                        - loop_penalty(
                            nxt,
                            beam["trajectory"],
                            args.backtrack_penalty,
                            args.revisit_penalty,
                            args.recent_loop_penalty,
                            args.anti_loop_window,
                        ),
                    }
                )
        if not expanded:
            break
        beams = sorted(
            expanded,
            key=lambda item: (
                item["current"] == goal,
                item["score"] - args.lookahead_distance_weight * distance_after_action(item["current"], "stop", distances),
                -distance_after_action(item["current"], "stop", distances),
            ),
            reverse=True,
        )[: max(1, args.lookahead_beam_size)]

    best = sorted(
        beams,
        key=lambda item: (
            item["current"] == goal,
            item["score"] - args.lookahead_distance_weight * distance_after_action(item["current"], "stop", distances),
            -distance_after_action(item["current"], "stop", distances),
        ),
        reverse=True,
    )[0]
    return str(best["first_action"])


@torch.no_grad()
def beam_rollout_one(
    sample: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[str], tuple[int, int], bool]:
    goal = tuple(sample["goal"])
    distances = shortest_distances(sample)
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else max(1, int(math.ceil(min(args.absolute_max_steps, args.length_factor * int(sample["shortest_path_length"])))))
    )
    beams = [
        {
            "current": tuple(sample["start"]),
            "actions": [],
            "score": 0.0,
            "trajectory": [tuple(sample["start"])],
            "stopped": False,
        }
    ]
    completed = []
    for _ in range(max_steps):
        expanded = []
        for beam in beams:
            current = beam["current"]
            if beam["stopped"] or current == goal:
                completed.append(beam)
                continue
            prompt = state_question(
                sample,
                [current[0], current[1]],
                include_distance_signals=args.include_distance_signals,
                include_valid_actions=args.include_valid_actions,
                include_history=args.include_history,
                history=beam["trajectory"],
                history_window=args.history_window,
                distances=distances if args.include_distance_signals else None,
            )
            candidates = candidate_set(sample, current, "mask_invalid_no_stop")
            scored = score_candidates(
                model,
                tokenizer,
                [prompt],
                [candidates],
                [current],
                [distances],
                args.target_format,
                device,
                args.max_source_length,
            )[0][: max(1, args.beam_branching)]
            recent = set(beam["trajectory"][-max(1, args.anti_loop_window) :])
            for action, logprob in scored:
                if action == "stop":
                    expanded.append({**beam, "score": beam["score"] + logprob, "stopped": True})
                    continue
                nxt = move(current, action)
                loop_penalty = args.anti_loop_penalty if args.beam_anti_loop and nxt in recent else 0.0
                executor_bonus = -args.beam_distance_weight * distance_after_action(current, action, distances)
                expanded.append(
                    {
                        "current": nxt,
                        "actions": [*beam["actions"], action],
                        "score": beam["score"] + logprob + executor_bonus - loop_penalty,
                        "trajectory": [*beam["trajectory"], nxt],
                        "stopped": False,
                    }
                )
        if not expanded:
            break
        beams = sorted(
            expanded,
            key=lambda item: (
                item["current"] == goal,
                item["score"],
                -distance_after_action(item["current"], "stop", distances),
            ),
            reverse=True,
        )[: max(1, args.beam_size)]
        if any(beam["current"] == goal for beam in beams):
            break
    candidates = [*completed, *beams]
    best = sorted(
        candidates,
        key=lambda item: (
            item["current"] == goal,
            item["score"],
            -distance_after_action(item["current"], "stop", distances),
        ),
        reverse=True,
    )[0]
    return best["actions"], best["current"], bool(best["stopped"] or best["current"] == goal)


@torch.no_grad()
def beam_no_revisit_rollout_one(
    sample: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[str], tuple[int, int], bool]:
    goal = tuple(sample["goal"])
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else max(1, int(math.ceil(min(args.absolute_max_steps, args.length_factor * int(sample["shortest_path_length"])))))
    )
    beams = [
        {
            "current": tuple(sample["start"]),
            "actions": [],
            "score": 0.0,
            "trajectory": [tuple(sample["start"])],
        }
    ]
    completed = []
    for _ in range(max_steps):
        if not beams:
            break
        completed.extend(beam for beam in beams if beam["current"] == goal)
        active_beams = [beam for beam in beams if beam["current"] != goal]
        if not active_beams:
            break

        prompts = [
            state_question(
                sample,
                [beam["current"][0], beam["current"][1]],
                include_distance_signals=args.include_distance_signals,
                include_valid_actions=args.include_valid_actions,
                include_history=args.include_history,
                history=beam["trajectory"],
                history_window=args.history_window,
                distances=None,
            )
            for beam in active_beams
        ]
        candidate_rows = []
        for beam in active_beams:
            visited = set(beam["trajectory"])
            candidates = []
            for action in ACTIONS:
                nxt = move(beam["current"], action)
                if nxt in visited:
                    continue
                if action in valid_actions(sample, beam["current"]):
                    candidates.append(action)
            candidate_rows.append(candidates)

        scored_rows = score_candidates(
            model,
            tokenizer,
            prompts,
            candidate_rows,
            [beam["current"] for beam in active_beams],
            [None for _ in active_beams],
            args.target_format,
            device,
            args.max_source_length,
        )
        expanded = []
        for beam, scored in zip(active_beams, scored_rows):
            for action, logprob in scored[: max(1, args.beam_branching)]:
                nxt = move(beam["current"], action)
                expanded.append(
                    {
                        "current": nxt,
                        "actions": [*beam["actions"], action],
                        "score": beam["score"] + logprob,
                        "trajectory": [*beam["trajectory"], nxt],
                    }
                )
        beams = sorted(expanded, key=lambda item: (item["current"] == goal, item["score"]), reverse=True)[
            : max(1, args.beam_size)
        ]
        if any(beam["current"] == goal for beam in beams):
            completed.extend(beam for beam in beams if beam["current"] == goal)
            break

    candidates = completed if completed else beams
    if not candidates:
        return [], tuple(sample["start"]), False
    best = sorted(candidates, key=lambda item: (item["current"] == goal, item["score"]), reverse=True)[0]
    return best["actions"], best["current"], best["current"] == goal


@torch.no_grad()
def beam_no_revisit_rollout_batch(
    samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> list[tuple[list[str], tuple[int, int], bool]]:
    goals = [tuple(sample["goal"]) for sample in samples]
    max_steps = [
        args.max_steps
        if args.max_steps is not None
        else max(1, int(math.ceil(min(args.absolute_max_steps, args.length_factor * int(sample["shortest_path_length"])))))
        for sample in samples
    ]
    beams_by_sample = [
        [
            {
                "current": tuple(sample["start"]),
                "actions": [],
                "score": 0.0,
                "trajectory": [tuple(sample["start"])],
            }
        ]
        for sample in samples
    ]
    completed_by_sample: list[list[dict[str, Any]]] = [[] for _ in samples]

    for step in range(max(max_steps, default=0)):
        prompts = []
        candidate_rows = []
        owner: list[int] = []
        source_beams = []
        next_beams_by_sample: list[list[dict[str, Any]]] = [[] for _ in samples]

        for sample_index, (sample, beams) in enumerate(zip(samples, beams_by_sample)):
            goal = goals[sample_index]
            for beam in beams:
                if beam["current"] == goal:
                    completed_by_sample[sample_index].append(beam)
                    continue
                if step >= int(max_steps[sample_index]):
                    next_beams_by_sample[sample_index].append(beam)
                    continue

                visited = set(beam["trajectory"])
                candidates = []
                for action in ACTIONS:
                    nxt = move(beam["current"], action)
                    if nxt in visited:
                        continue
                    if action in valid_actions(sample, beam["current"]):
                        candidates.append(action)
                if not candidates:
                    continue

                prompts.append(
                    state_question(
                        sample,
                        [beam["current"][0], beam["current"][1]],
                        include_distance_signals=args.include_distance_signals,
                        include_valid_actions=args.include_valid_actions,
                        include_history=args.include_history,
                        history=beam["trajectory"],
                        history_window=args.history_window,
                        distances=None,
                    )
                )
                candidate_rows.append(candidates)
                owner.append(sample_index)
                source_beams.append(beam)

        if not prompts:
            beams_by_sample = next_beams_by_sample
            break

        scored_rows = score_candidates(
            model,
            tokenizer,
            prompts,
            candidate_rows,
            [beam["current"] for beam in source_beams],
            [None for _ in source_beams],
            args.target_format,
            device,
            args.max_source_length,
        )
        for sample_index, beam, scored in zip(owner, source_beams, scored_rows):
            for action, logprob in scored[: max(1, args.beam_branching)]:
                nxt = move(beam["current"], action)
                next_beams_by_sample[sample_index].append(
                    {
                        "current": nxt,
                        "actions": [*beam["actions"], action],
                        "score": beam["score"] + logprob,
                        "trajectory": [*beam["trajectory"], nxt],
                    }
                )

        beams_by_sample = [
            sorted(beams, key=lambda item: (item["current"] == goals[index], item["score"]), reverse=True)[
                : max(1, args.beam_size)
            ]
            for index, beams in enumerate(next_beams_by_sample)
        ]
        if all(completed_by_sample[index] or not beams_by_sample[index] for index in range(len(samples))):
            break

    results = []
    for sample, goal, completed, beams in zip(samples, goals, completed_by_sample, beams_by_sample):
        candidates = completed if completed else beams
        if not candidates:
            results.append(([], tuple(sample["start"]), False))
            continue
        best = sorted(candidates, key=lambda item: (item["current"] == goal, item["score"]), reverse=True)[0]
        results.append((best["actions"], best["current"], best["current"] == goal))
    return results


@torch.no_grad()
def rollout(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run stepwise decoding over a dataset and return prediction rows."""
    device = select_device()
    samples = [ensure_grid_size(sample) for sample in load_original_samples(args.dataset, args.max_samples)]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True).to(device)
    model.eval()

    rows = []
    if args.mode in {"beam", "beam_no_revisit"}:
        for start in tqdm(range(0, len(samples), args.batch_size), desc=args.mode):
            batch = samples[start : start + args.batch_size]
            if args.mode == "beam_no_revisit":
                batch_results = beam_no_revisit_rollout_batch(batch, model, tokenizer, device, args)
            else:
                batch_results = [beam_rollout_one(sample, model, tokenizer, device, args) for sample in batch]
            for offset, (sample, result) in enumerate(zip(batch, batch_results)):
                sample_actions, final, stopped = result
                text = " ".join(sample_actions)
                row = {
                    "id": sample.get("id", sample.get("example_id", start + offset)),
                    "english": sample.get("nl_description", sample.get("question", "")),
                    "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
                    "generated": text,
                    "model_output": text,
                    "rollout_mode": args.mode,
                    "rollout_steps": len(sample_actions),
                    "rollout_final": list(final),
                    "rollout_stopped": stopped,
                }
                for key in ("world", "start", "goal", "goals", "obstacles", "shortest_path_length", "grid_size"):
                    if key in sample:
                        row[key] = sample[key]
                rows.append(row)
        return rows

    for start in tqdm(range(0, len(samples), args.batch_size), desc=args.mode):
        batch = samples[start : start + args.batch_size]
        currents = [tuple(sample["start"]) for sample in batch]
        goals = [tuple(sample["goal"]) for sample in batch]
        needs_distances = args.include_distance_signals or args.target_format != "action" or args.mode in {"topk_verifier", "topk_lookahead"}
        distance_maps = [shortest_distances(sample) if needs_distances else None for sample in batch]
        actions: list[list[str]] = [[] for _ in batch]
        trajectories: list[list[tuple[int, int]]] = [[tuple(sample["start"])] for sample in batch]
        stopped = [False for _ in batch]
        max_steps = [
            args.max_steps
            if args.max_steps is not None
            else max(1, int(math.ceil(min(args.absolute_max_steps, args.length_factor * int(sample["shortest_path_length"])))))
            for sample in batch
        ]

        for step in range(max(max_steps)):
            active = [i for i, done in enumerate(stopped) if not done and step < max_steps[i]]
            if not active:
                break
            prompts = [
                state_question(
                    batch[i],
                    [currents[i][0], currents[i][1]],
                    include_distance_signals=args.include_distance_signals,
                    include_valid_actions=args.include_valid_actions,
                    include_history=args.include_history,
                    history=trajectories[i],
                    history_window=args.history_window,
                    distances=distance_maps[i],
                )
                for i in active
            ]
            if args.mode == "plain":
                chosen = greedy_actions(model, tokenizer, prompts, device, args.max_source_length)
            elif args.mode == "topk_lookahead":
                chosen = [
                    lookahead_action_one(
                        batch[i],
                        currents[i],
                        trajectories[i],
                        model,
                        tokenizer,
                        device,
                        args,
                        distance_maps[i],
                    )
                    for i in active
                ]
            else:
                candidates = [candidate_set(batch[i], currents[i], args.mode) for i in active]
                scored = score_candidates(
                    model,
                    tokenizer,
                    prompts,
                    candidates,
                    [currents[i] for i in active],
                    [distance_maps[i] for i in active],
                    args.target_format,
                    device,
                    args.max_source_length,
                )
                chosen = [
                    select_scored_action(
                        row_scores,
                        args.mode,
                        currents[i],
                        distance_maps[i],
                        trajectories[i],
                        args.top_k,
                        args.anti_loop_window,
                        args.anti_loop_penalty,
                    )
                    for i, row_scores in zip(active, scored)
                ]

            for local_index, action in zip(active, chosen):
                if currents[local_index] == goals[local_index] or action == "stop":
                    stopped[local_index] = True
                    continue
                actions[local_index].append(action)
                currents[local_index] = move(currents[local_index], action)
                trajectories[local_index].append(currents[local_index])

        for offset, sample in enumerate(batch):
            text = " ".join(actions[offset])
            row = {
                "id": sample.get("id", sample.get("example_id", start + offset)),
                "english": sample.get("nl_description", sample.get("question", "")),
                "ground_truth": sample.get("agent_as_a_point", sample.get("target", "")),
                "generated": text,
                "model_output": text,
                "rollout_mode": args.mode,
                "rollout_steps": len(actions[offset]),
                "rollout_final": list(currents[offset]),
                "rollout_stopped": stopped[offset],
            }
            for key in ("world", "start", "goal", "goals", "obstacles", "shortest_path_length", "grid_size"):
                if key in sample:
                    row[key] = sample[key]
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a stepwise policy.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=[
            "plain",
            "mask_invalid",
            "mask_invalid_no_stop",
            "topk_verifier",
            "topk_lookahead",
            "beam",
            "beam_no_revisit",
            "anti_loop",
        ],
        default="mask_invalid_no_stop",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--length-factor", type=float, default=3.0)
    parser.add_argument("--absolute-max-steps", type=int, default=96)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-distance-signals", action="store_true")
    parser.add_argument("--include-valid-actions", action="store_true")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--target-format", choices=["action", "action_next_distance"], default="action")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--anti-loop-window", type=int, default=4)
    parser.add_argument("--anti-loop-penalty", type=float, default=2.0)
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--lookahead-beam-size", type=int, default=4)
    parser.add_argument("--lookahead-distance-weight", type=float, default=1.0)
    parser.add_argument("--lookahead-model-weight", type=float, default=1.0)
    parser.add_argument("--backtrack-penalty", type=float, default=4.0)
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--recent-loop-penalty", type=float, default=2.0)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--beam-branching", type=int, default=4)
    parser.add_argument("--beam-distance-weight", type=float, default=0.15)
    parser.add_argument("--beam-anti-loop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = rollout(args)
    write_jsonl(args.out, rows)
    print(json.dumps({"wrote": str(args.out), "num_predictions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

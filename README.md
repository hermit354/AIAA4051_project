# Stepwise Decoding for Grid Path Planning

## Project Overview
This project studies language-model path planning on single-goal grid worlds. Each input describes a grid, a start cell, a goal cell, and obstacles. The model outputs an action sequence using `up`, `down`, `left`, and `right`. The main question is whether a language model can produce feasible and optimal paths, especially when test grids are larger, denser, or shaped differently from the training grids.

The codebase includes data generation, seq2seq training, prompting, simulator-controlled stepwise rollout, evaluation, and figure-generation components.

## Motivation
Direct sequence generation often works on in-distribution grids but degrades on longer or denser layouts. A small textual mistake can make the whole path invalid. This project compares direct decoding with stepwise methods that repeatedly query a model for the next action and use an executor to check validity, distance-to-goal progress, and loop behavior.

The key gap is between text exact match and executable planning quality. The project therefore evaluates predictions by actually executing them in the grid, not only by string comparison.

## System Architecture
The pipeline has five stages:

1. **Data generation**: `src/data/generate_phase0.py` creates grid-world train, validation, ID test, and OOD test splits.
2. **Dataset variants**: `src/data/make_*_datasets.py` converts the base data into input-format, SFT-method, DAgger, executor-feedback, and stepwise-policy datasets.
3. **Modeling**: `src/models/train_seq2seq.py` fine-tunes local seq2seq models; `src/models/infer_*.py` runs direct prompting or decoding ablations.
4. **Stepwise executor rollout**: `src/models/rollout_stepwise_policy.py` queries a one-step policy, filters candidates with simulator constraints, and optionally uses top-k lookahead.
5. **Evaluation and reporting**: `src/eval/evaluate_predictions.py` executes predicted plans and reports success, feasibility, optimality, exact match, and failure types. Summary scripts aggregate results for figures and tables.

Data flows from JSON/JSONL datasets into model runners, then into prediction JSONL files, then into evaluator JSON/CSV summaries. The executor is the central bridge between model text and planning behavior: it turns action strings into trajectories and failure categories.

## Phase Roadmap
The repository is organized around three experimental phases. Each phase has a distinct purpose and produces artifacts used by later phases.

| Phase | Purpose | Main inputs | Main outputs | Key entry points |
| --- | --- | --- | --- | --- |
| Phase 0: Dataset construction | Build the base grid-world benchmark and verify split quality. | Grid specifications, obstacle settings, split definitions, and random seed. | `data/generated/phase0` train/validation, ID test, OOD size, OOD density, and OOD aspect-ratio splits. | `src/data/generate_phase0.py`, `src/data/validate_phase0.py` |
| Phase 1: Direct baselines | Establish direct sequence-generation baselines on the Phase 0 benchmark. | Phase 0 JSONL splits and local seq2seq base models. | Baseline checkpoints, prediction JSONL files, per-split metrics, and grouped summary CSV files under `outputs/phase1_*`. | `src/models/run_phase1_baselines.py`, `scripts/run_phase1_baselines_gpu5.sh`, `src/eval/summarize_phase1.py` |
| Phase 2: Method improvements and ablations | Test whether representation changes, supervision changes, prompting, and stepwise executor rollout improve executable planning. | Phase 0 records, Phase 2 converted datasets, Phase 1 checkpoints, and evaluated rollout traces. | Method-specific datasets, trained checkpoints, rollout predictions, failure analyses, and compact summaries. | `src/data/make_*_datasets.py`, `src/models/run_phase2_*.py`, `src/models/rollout_stepwise_policy.py`, `scripts/run_phase2_*.sh` |

Phase 0 is the data foundation. It creates the common task format and the split structure used by every later experiment. The generated examples include the grid, start and goal coordinates, obstacles, natural or structured input text, target actions, and shortest-path metadata.

Phase 1 measures how far standard direct decoding can go before adding extra planning structure. These runs train seq2seq models to emit the full path in one shot, evaluate each output with the executor, and summarize where direct generation succeeds or fails across ID and OOD splits.

Phase 2 contains the main experimental variations. It converts Phase 0 data into alternative input formats, target formats, data-scaling subsets, stepwise-policy examples, DAgger correction data, executor-feedback data, and prompting setups. The most important Phase 2 path is the stepwise executor workflow: a model predicts one action at a time, the simulator filters or scores candidate moves, and evaluation checks the completed trajectory.

## Key Modules
| File | Role |
| --- | --- |
| `src/data/generate_phase0.py` | Generates the base grid-world datasets and shortest-path targets. |
| `src/data/make_stepwise_policy_dataset.py` | Converts full-path examples into state-to-next-action supervision. |
| `src/models/train_seq2seq.py` | Fine-tunes seq2seq planners and records validation/checkpoint metrics. |
| `src/models/infer_prompting.py` | Runs prompt-only local-model baselines. |
| `src/models/infer_decoding_ablation.py` | Compares greedy, beam, constrained, reranked, and state-aware decoding. |
| `src/models/rollout_stepwise_policy.py` | Runs simulator-controlled stepwise decoding and lookahead variants. |
| `src/eval/evaluate_predictions.py` | Executes predicted action sequences and computes planning metrics. |
| `src/eval/summarize_*.py` | Aggregates per-split metrics into report-ready CSV summaries. |

## Repository Structure
```text
.
├── src/
│   ├── data/        # Dataset generation, conversion, and failure-analysis utilities
│   ├── models/      # Training, inference, model registry, and stepwise rollout
│   └── eval/        # Executable path evaluation and result summarization
├── scripts/         # Reproducible experiment launchers and figure scripts
├── data/samples/    # Minimal examples for quick evaluator checks
├── results/         # Lightweight result summaries
├── outputs/         # Local experiment outputs; most files are ignored
├── .gitignore       # Prevents models, checkpoints, logs, secrets, and large data from being committed
└── requirements.txt # Minimal Python dependencies
```

Large local directories such as `source_models/`, full `outputs/`, checkpoints, logs, raw datasets, report drafts, LaTeX files, and poster assets are intentionally excluded from version control.

For a detailed end-to-end map of every code module, dataset location, result artifact, and experiment launcher, see [`WORKFLOW.md`](WORKFLOW.md).

## Installation
Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU training requires a PyTorch build compatible with your CUDA version. Full experiments also require local Hugging Face model directories or downloadable model names.

## How to Run
Run the quick evaluator sample:

```bash
python src/eval/evaluate_predictions.py \
  --predictions data/samples/sample_predictions.jsonl \
  --dataset data/samples/path_planning_sample.jsonl
```

Run evaluator sanity checks:

```bash
python src/eval/sanity_check.py \
  --metrics-out outputs/evaluator_sanity_metrics.json \
  --traces-out outputs/predictions/evaluator_sanity_cases.jsonl
```

Generate the base Phase 0 dataset:

```bash
python src/data/generate_phase0.py --output-dir data/generated/phase0 --seed 1
```

Train a seq2seq model from a config:

```bash
python src/models/train_seq2seq.py --config path/to/train_config.json
```

Run stepwise rollout and evaluation. This command requires a trained checkpoint and generated evaluation split:

```bash
python src/models/rollout_stepwise_policy.py \
  --checkpoint outputs/phase2_size_generalization/stepwise_policy/flan-t5-base/best \
  --dataset data/generated/phase2_size_generalization/stepwise_policy/jsonl/test_size_12x12.jsonl \
  --out outputs/example_stepwise_predictions.jsonl \
  --mode topk_lookahead \
  --include-distance-signals

python src/eval/evaluate_predictions.py \
  --predictions outputs/example_stepwise_predictions.jsonl \
  --dataset data/generated/phase2_size_generalization/stepwise_policy/jsonl/test_size_12x12.jsonl \
  --metrics-out outputs/example_stepwise_metrics.json
```

Shell launchers in `scripts/` wrap the same Python entry points. They infer `PROJECT_ROOT` from the repository location and allow `ENV_NAME`, `CONDA_BASE`, `GPU_ID`, `DATA_ROOT`, and `OUTPUT_ROOT` to be overridden.

## Experiment Scripts
| Script | Purpose | Main prerequisites |
| --- | --- | --- |
| `scripts/run_phase1_baselines_gpu5.sh` | Train and evaluate direct seq2seq baselines. | Generated Phase 0 data and local base models. |
| `scripts/run_phase2_prompting.sh` | Run local prompt-only baselines. | Local model directories and generated evaluation splits. |
| `scripts/run_phase2_sft_methods.sh` | Compare supervised fine-tuning target formats. | Converted Phase 2 SFT datasets. |
| `scripts/run_decoding_ablation.sh` | Compare decoding strategies for a trained checkpoint. | A seq2seq checkpoint and evaluation splits. |
| `scripts/run_phase2_stepwise_policy.sh` | Train the one-step policy used by rollout. | Stepwise-policy JSONL data. |
| `scripts/run_phase2_stepwise_eval_suite.sh` | Evaluate stepwise rollout modes on selected splits. | A trained one-step checkpoint and `DATA_ROOT`. |
| `scripts/run_phase2_stepwise_dagger.sh` | Build and train DAgger-style correction data. | Previous rollout outputs with evaluated traces. |
| `scripts/run_phase2_deepseek_prompting.sh` | Run API prompting baselines. | `DS_API` environment variable and generated evaluation splits. |

## Configuration
Most experiments are configured through command-line flags or generated JSON config files.

Important parameters:

- `model_name_or_path`: local model directory or Hugging Face model name.
- `train_path` / `eval_path`: JSONL files used by seq2seq training.
- `max_source_length` / `max_target_length`: tokenizer truncation limits.
- `per_device_train_batch_size` / `gradient_accumulation_steps`: effective training batch control.
- `selection_metric`: validation metric used to keep the best checkpoint.
- `mode` in stepwise rollout: one of direct, invalid-action masking, top-k verifier, lookahead, or loop-aware variants.
- `length_factor` and `absolute_max_steps`: rollout safety bounds that prevent unbounded loops.

DeepSeek/API prompting uses an environment variable such as `DS_API`. API keys must stay outside the repository.

## Input and Output Format
Input data is JSONL. Each row contains grid metadata, a natural-language or structured question, a target action sequence, and an executable `world` matrix.

```json
{
  "id": "sample_easy_success",
  "grid_size": [4, 4],
  "start": [0, 0],
  "goal": [0, 3],
  "obstacles": [],
  "question": "Grid size: 4 by 4. Start: (0,0). Goal: (0,3). Obstacles: []",
  "target": "right right right",
  "world": [[2, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}
```

Prediction files can contain only `id` and `model_output` when a dataset file is passed separately:

```json
{"id": "sample_easy_success", "model_output": "right right right"}
```

Evaluation outputs include aggregate metrics and optional per-example traces.

## Evaluation
The evaluator executes each predicted action sequence in the grid. It reports:

- `success_rate`: final position reaches the goal.
- `feasibility`: no invalid token, obstacle hit, or out-of-bounds move.
- `optimality`: success with shortest-path length.
- `exact_match`: normalized text match with the reference action string.
- `failure_type`: first meaningful failure category, such as `INVALID_TOKEN`, `HIT_OBSTACLE`, `PREMATURE_STOP`, or `LOOP_TOO_LONG`.

Execution-based metrics are appropriate because the task is planning, not only text generation. Two different action strings can both solve the task, and a string with high token overlap can still be invalid.

## Design Choices
- The repository keeps `data`, `models`, and `eval` separate so that generation, learning, and measurement are easy to explain independently.
- The executor is shared between training-time metrics and final evaluation to keep success definitions consistent.
- Stepwise rollout uses simulator feedback because it can prevent invalid moves and expose failure modes that direct text generation hides.
- Summary artifacts are separated from full outputs so the repository stays readable and lightweight.

## Limitations and Future Work
- Full training requires GPU resources and local model downloads that are not committed to version control.
- Large-grid and dense-grid generalization remains difficult, especially when paths are long or require detours.
- API prompting was explored but should be treated separately because it depends on external service behavior and keys.
- Future work could add stronger search, learned value functions, richer curriculum generation, and a smaller end-to-end demo model.

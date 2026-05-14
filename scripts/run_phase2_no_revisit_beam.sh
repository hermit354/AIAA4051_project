#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_size_generalization/no_revisit_beam}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_size_generalization/no_revisit_beam_stepwise}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_size_generalization/no_revisit_beam/t5-base}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/source_models/t5-base}"
TEST_SPLITS="${TEST_SPLITS:-test_size_8x8 test_size_9x9 test_size_12x12 test_size_14x14 test_size_16x16}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-224}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

if [ ! -f "${RAW_DATA_ROOT}/manifest.json" ]; then
  python src/data/generate_no_revisit_beam_data.py --output-dir "${RAW_DATA_ROOT}"
fi

if [ ! -f "${DATA_ROOT}/manifest.json" ]; then
  python src/data/make_stepwise_policy_dataset.py \
    --source-dir "${RAW_DATA_ROOT}/jsonl" \
    --output-dir "${DATA_ROOT}"
fi

python src/models/check_compute.py \
  --report-out "${OUTPUT_ROOT}/compute_check.json" \
  --require-free-gpu

cat > "${OUTPUT_ROOT}/train_config.json" <<JSON
{
  "model_name_or_path": "${MODEL_PATH}",
  "output_dir": "${OUTPUT_ROOT}",
  "train_path": "${DATA_ROOT}/jsonl/train.jsonl",
  "eval_path": "${DATA_ROOT}/jsonl/val.jsonl",
  "max_source_length": ${MAX_SOURCE_LENGTH},
  "max_target_length": 8,
  "num_train_epochs": 20,
  "per_device_train_batch_size": 32,
  "per_device_eval_batch_size": 64,
  "gradient_accumulation_steps": 1,
  "learning_rate": 5e-5,
  "warmup_ratio": 0.03,
  "eval_steps": 1000,
  "save_steps": 1000,
  "save_total_limit": 3,
  "early_stopping_patience": 8,
  "eval_max_new_tokens": 4,
  "num_beams": 1,
  "seed": 1,
  "fp16": false,
  "bf16": true,
  "selection_metric": "text_exact_match"
}
JSON

if [ ! -d "${OUTPUT_ROOT}/best" ] || [ ! -f "${OUTPUT_ROOT}/train_summary.json" ]; then
  python src/models/train_seq2seq.py --config "${OUTPUT_ROOT}/train_config.json"
fi

for split in ${TEST_SPLITS}; do
  split_out="${OUTPUT_ROOT}/beam_no_revisit/${split}"
  metrics="${split_out}/metrics.json"
  if [ -f "${metrics}" ]; then
    echo "skip beam_no_revisit/${split}: metrics exists"
    continue
  fi
  mkdir -p "${split_out}"
  python src/models/rollout_stepwise_policy.py \
    --checkpoint "${OUTPUT_ROOT}/best" \
    --dataset "${RAW_DATA_ROOT}/jsonl/${split}.jsonl" \
    --out "${split_out}/predictions.jsonl" \
    --mode beam_no_revisit \
    --batch-size 32 \
    --max-source-length "${MAX_SOURCE_LENGTH}" \
    --beam-size 4 \
    --beam-branching 4 \
    --length-factor 4 \
    --absolute-max-steps 128
  python src/eval/evaluate_predictions.py \
    --predictions "${split_out}/predictions.jsonl" \
    --dataset "${RAW_DATA_ROOT}/jsonl/${split}.jsonl" \
    --metrics-out "${metrics}" \
    --predictions-out "${split_out}/evaluated_predictions.jsonl" \
    --failure-breakdown-out "${split_out}/failure_breakdown.json"
done

python src/data/analyze_stepwise_failures.py \
  --root "${OUTPUT_ROOT}/beam_no_revisit" \
  --splits ${TEST_SPLITS} \
  --out "${OUTPUT_ROOT}/beam_no_revisit/failure_distance_summary.json"

#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-6}"
VARIANT_NAME="${VARIANT_NAME:?set VARIANT_NAME}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_size_generalization/${VARIANT_NAME}}"
ORIGINAL_DATA_ROOT="${ORIGINAL_DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_size_generalization/react_trace_compact_sizegen/jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_size_generalization/${VARIANT_NAME}/flan-t5-base}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/source_models/flan-t5-base}"
TEST_SPLITS="${TEST_SPLITS:-test_size_8x8 test_size_9x9 test_size_12x12 test_size_14x14 test_size_16x16}"
MODES="${MODES:-mask_invalid_no_stop topk_verifier anti_loop beam}"
DATA_FLAGS="${DATA_FLAGS:-}"
ROLLOUT_FLAGS="${ROLLOUT_FLAGS:-}"
TARGET_FORMAT="${TARGET_FORMAT:-action}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-224}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-8}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

if [ ! -f "${DATA_ROOT}/manifest.json" ]; then
  python src/data/make_stepwise_policy_dataset.py \
    --source-dir "${ORIGINAL_DATA_ROOT}" \
    --output-dir "${DATA_ROOT}" \
    --target-format "${TARGET_FORMAT}" \
    ${DATA_FLAGS}
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
  "max_target_length": ${MAX_TARGET_LENGTH},
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
  "eval_max_new_tokens": ${EVAL_MAX_NEW_TOKENS},
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

for mode in ${MODES}; do
  for split in ${TEST_SPLITS}; do
    split_out="${OUTPUT_ROOT}/${mode}/${split}"
    metrics="${split_out}/metrics.json"
    if [ -f "${metrics}" ]; then
      echo "skip ${mode}/${split}: metrics exists"
      continue
    fi
    mkdir -p "${split_out}"
    python src/models/rollout_stepwise_policy.py \
      --checkpoint "${OUTPUT_ROOT}/best" \
      --dataset "${ORIGINAL_DATA_ROOT}/${split}.jsonl" \
      --out "${split_out}/predictions.jsonl" \
      --mode "${mode}" \
      --batch-size 16 \
      --max-source-length "${MAX_SOURCE_LENGTH}" \
      --target-format "${TARGET_FORMAT}" \
      ${ROLLOUT_FLAGS}
    python src/eval/evaluate_predictions.py \
      --predictions "${split_out}/predictions.jsonl" \
      --dataset "${ORIGINAL_DATA_ROOT}/${split}.jsonl" \
      --metrics-out "${metrics}" \
      --predictions-out "${split_out}/evaluated_predictions.jsonl" \
      --failure-breakdown-out "${split_out}/failure_breakdown.json"
  done
done

if [ -d "${OUTPUT_ROOT}/mask_invalid_no_stop" ]; then
  python src/data/analyze_stepwise_failures.py \
    --root "${OUTPUT_ROOT}/mask_invalid_no_stop" \
    --splits ${TEST_SPLITS} \
    --out "${OUTPUT_ROOT}/mask_invalid_no_stop/failure_distance_summary.json"
fi

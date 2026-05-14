#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
TEST_SPLITS="${TEST_SPLITS:?set TEST_SPLITS}"
MODES="${MODES:-topk_lookahead}"
INCLUDE_HISTORY="${INCLUDE_HISTORY:-false}"
INCLUDE_DISTANCE="${INCLUDE_DISTANCE:-true}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-256}"
TARGET_FORMAT="${TARGET_FORMAT:-action_next_distance}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
LENGTH_FACTOR="${LENGTH_FACTOR:-3.0}"
ABSOLUTE_MAX_STEPS="${ABSOLUTE_MAX_STEPS:-96}"
CHECK_MAX_USED_MB="${CHECK_MAX_USED_MB:-500}"
CHECK_MAX_UTILIZATION="${CHECK_MAX_UTILIZATION:-5}"
TOP_K="${TOP_K:-3}"
LOOKAHEAD_STEPS="${LOOKAHEAD_STEPS:-3}"
LOOKAHEAD_BEAM_SIZE="${LOOKAHEAD_BEAM_SIZE:-4}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/models/check_compute.py \
  --report-out "${OUTPUT_ROOT}/compute_check_${RUN_NAME}.json" \
  --max-used-mb "${CHECK_MAX_USED_MB}" \
  --max-utilization "${CHECK_MAX_UTILIZATION}" \
  --require-free-gpu

history_arg=()
if [ "${INCLUDE_HISTORY}" = "true" ]; then
  history_arg=(--include-history --history-window 4)
fi
distance_arg=()
if [ "${INCLUDE_DISTANCE}" = "true" ]; then
  distance_arg=(--include-distance-signals)
fi

for mode in ${MODES}; do
  for split in ${TEST_SPLITS}; do
    split_out="${OUTPUT_ROOT}/${mode}/${split}"
    metrics="${split_out}/metrics.json"
    if [ -f "${metrics}" ]; then
      echo "skip ${RUN_NAME}/${mode}/${split}: metrics exists"
      continue
    fi
    mkdir -p "${split_out}"
    python src/models/rollout_stepwise_policy.py \
      --checkpoint "${CHECKPOINT}" \
      --dataset "${DATA_ROOT}/${split}.jsonl" \
      --out "${split_out}/predictions.jsonl" \
      --mode "${mode}" \
      --batch-size "${ROLLOUT_BATCH_SIZE}" \
      --max-source-length "${MAX_SOURCE_LENGTH}" \
      --target-format "${TARGET_FORMAT}" \
      "${distance_arg[@]}" \
      "${history_arg[@]}" \
      --length-factor "${LENGTH_FACTOR}" \
      --absolute-max-steps "${ABSOLUTE_MAX_STEPS}" \
      --top-k "${TOP_K}" \
      --lookahead-steps "${LOOKAHEAD_STEPS}" \
      --lookahead-beam-size "${LOOKAHEAD_BEAM_SIZE}"
    python src/eval/evaluate_predictions.py \
      --predictions "${split_out}/predictions.jsonl" \
      --dataset "${DATA_ROOT}/${split}.jsonl" \
      --metrics-out "${metrics}" \
      --predictions-out "${split_out}/evaluated_predictions.jsonl" \
      --failure-breakdown-out "${split_out}/failure_breakdown.json"
  done
done

#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
SPLIT="${SPLIT:?set SPLIT}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CHECKPOINT="${PROJECT_ROOT}/outputs/phase2_size_generalization/stepwise_progress_target/flan-t5-base/best"
DATASET="${PROJECT_ROOT}/data/generated/phase2_size_generalization/react_trace_compact_sizegen/jsonl/${SPLIT}.jsonl"
OUT_DIR="${PROJECT_ROOT}/outputs/phase2_size_generalization/stepwise_progress_target/flan-t5-base/topk_lookahead/${SPLIT}"
mkdir -p "${OUT_DIR}"

python src/models/rollout_stepwise_policy.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --out "${OUT_DIR}/predictions.jsonl" \
  --mode topk_lookahead \
  --batch-size 8 \
  --max-source-length 192 \
  --target-format action_next_distance \
  --include-distance-signals \
  --top-k 3 \
  --lookahead-steps 3 \
  --lookahead-beam-size 4

python src/eval/evaluate_predictions.py \
  --predictions "${OUT_DIR}/predictions.jsonl" \
  --dataset "${DATASET}" \
  --metrics-out "${OUT_DIR}/metrics.json" \
  --predictions-out "${OUT_DIR}/evaluated_predictions.jsonl" \
  --failure-breakdown-out "${OUT_DIR}/failure_breakdown.json"

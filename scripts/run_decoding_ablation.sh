#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/phase1_baselines_flan_bf16/flan-t5-base/best}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/generated/phase0/jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_decoding/flan-t5-base}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

STRATEGIES=("$@")
if [ "${#STRATEGIES[@]}" -eq 0 ]; then
  STRATEGIES=(greedy beam4 beam8 token_constrained postprocess rerank_beam8 state_aware)
fi

SPLITS=(
  test_unseen_placement
  test_unseen_environment
  ood_size_5x5
  ood_size_7x7
  ood_size_8x8
  ood_size_9x9
  ood_size_10x10
  ood_dense_6x6
  ood_dense_5x5
  ood_dense_7x7
  ood_aspect_10x5
  ood_aspect_6x9
)

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/models/check_compute.py \
  --report-out "${OUTPUT_ROOT}/compute_check_gpu${GPU_ID}.json" \
  --require-free-gpu

for strategy in "${STRATEGIES[@]}"; do
  for split in "${SPLITS[@]}"; do
    split_out="${OUTPUT_ROOT}/${strategy}/${split}"
    predictions="${split_out}/predictions.jsonl"
    metrics="${split_out}/metrics.json"
    if [ -f "${metrics}" ]; then
      echo "skip ${strategy}/${split}: metrics already exists"
      continue
    fi
    mkdir -p "${split_out}"
    extra=()
    if [ -n "${MAX_SAMPLES}" ]; then
      extra+=(--max-samples "${MAX_SAMPLES}")
    fi
    python src/models/infer_decoding_ablation.py \
      --checkpoint "${CHECKPOINT}" \
      --dataset "${DATASET_DIR}/${split}.jsonl" \
      --out "${predictions}" \
      --strategy "${strategy}" \
      --batch-size 16 \
      --max-new-tokens 96 \
      "${extra[@]}"
    python src/eval/evaluate_predictions.py \
      --predictions "${predictions}" \
      --dataset "${DATASET_DIR}/${split}.jsonl" \
      --metrics-out "${metrics}" \
      --predictions-out "${split_out}/evaluated_predictions.jsonl" \
      --failure-breakdown-out "${split_out}/failure_breakdown.json"
  done
done


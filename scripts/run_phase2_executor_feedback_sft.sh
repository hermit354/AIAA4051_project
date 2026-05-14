#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/data/generated/phase0/jsonl}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/generated/phase2_sft_method/executor_feedback}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_sft_method}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/phase1_baselines_flan_bf16/flan-t5-base/best}"
MODEL="${MODEL:-flan-t5-base}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

if [ ! -f "${DATA_DIR}/jsonl/train.jsonl" ] || [ ! -f "${DATA_DIR}/jsonl/val.jsonl" ]; then
  python src/data/make_executor_feedback_sft.py \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${DATA_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --batch-size 16 \
    --num-beams 8
else
  echo "Executor-feedback dataset already exists at ${DATA_DIR}; skipping generation."
fi

python src/models/run_phase2_sft_methods.py \
  --data-root "${PROJECT_ROOT}/data/generated/phase2_sft_method" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}" \
  --methods executor_feedback \
  --allow-cpu

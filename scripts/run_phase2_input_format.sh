#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
MODEL="${MODEL:-flan-t5-base}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/generated/phase2_input_format}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_input_format}"
FORMATS=("$@")

if [ "${#FORMATS[@]}" -eq 0 ]; then
  FORMATS=(structured grid_matrix hybrid)
fi

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/models/check_compute.py \
  --report-out "${OUTPUT_ROOT}/compute_check_gpu${GPU_ID}.json" \
  --require-free-gpu

python src/models/run_phase2_input_format.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}" \
  --formats "${FORMATS[@]}"

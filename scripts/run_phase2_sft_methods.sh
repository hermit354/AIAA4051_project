#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-4}"
METHODS="${METHODS:-coordinate_trace multi_template}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_sft_method}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_sft_method}"
MODEL="${MODEL:-flan-t5-base}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/models/run_phase2_sft_methods.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}" \
  --methods ${METHODS} \
  --allow-cpu

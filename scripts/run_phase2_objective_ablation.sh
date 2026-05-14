#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-3}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/data/generated/phase0/jsonl}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_objective_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_objective_ablation}"
METHODS="${METHODS:-multi_reference auxiliary_mixture}"
MODEL="${MODEL:-flan-t5-base}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/data/make_objective_ablation_datasets.py \
  --source-dir "${SOURCE_DIR}" \
  --output-root "${DATA_ROOT}"

python src/models/run_phase2_sft_methods.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}" \
  --methods ${METHODS} \
  --allow-cpu

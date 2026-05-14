#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/generated/phase2_multiscale_sft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_sft_method/multiscale}"
MODEL="${MODEL:-flan-t5-base}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/models/check_compute.py \
  --report-out "${OUTPUT_ROOT}/compute_check_gpu${GPU_ID}.json"

python src/models/run_phase1_baselines.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --models "${MODEL}" \
  --test-splits \
    test_unseen_placement \
    test_unseen_environment \
    ood_size_5x5 \
    ood_size_7x7 \
    ood_size_8x8 \
    ood_size_9x9 \
    ood_size_10x10 \
    ood_dense_6x6 \
    ood_dense_5x5 \
    ood_dense_7x7 \
    ood_aspect_10x5 \
    ood_aspect_6x9 \
  --allow-cpu

#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase1_baselines_converged}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/generated/phase0}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/run_gpu${GPU_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
LOCK_FILE="${OUTPUT_ROOT}/phase1_gpu${GPU_ID}.lock"

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
  MODELS=(t5-small t5-base flan-t5-small flan-t5-base bart-base)
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [ -e "${LOCK_FILE}" ]; then
  old_pid="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Another Phase 1 baseline run appears active with PID ${old_pid}."
    echo "Lock file: ${LOCK_FILE}"
    exit 1
  fi
  rm -f "${LOCK_FILE}"
fi

echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

{
  echo "Started: $(date -u)"
  echo "Project: ${PROJECT_ROOT}"
  echo "Dataset: ${DATASET_DIR}"
  echo "Output: ${OUTPUT_ROOT}"
  echo "GPU_ID: ${GPU_ID}"
  echo "Models: ${MODELS[*]}"
  python src/models/check_compute.py \
    --report-out "${OUTPUT_ROOT}/compute_check_gpu${GPU_ID}.json" \
    --require-free-gpu
  python src/models/check_local_models.py \
    --models "${MODELS[@]}" \
    --report-out "${OUTPUT_ROOT}/local_model_integrity.json"
  python src/data/validate_phase0.py \
    --dataset-dir "${DATASET_DIR}" \
    --report-out "${OUTPUT_ROOT}/phase0_validation_strict.json"
  python src/models/run_phase1_baselines.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --models "${MODELS[@]}"
  echo "Completed: $(date -u)"
} 2>&1 | tee -a "${LOG_FILE}"


#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-5}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/data/generated/phase0/jsonl}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/generated/phase2_scaling/data_scaling}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_scaling/data_scaling}"
PERCENTS="${PERCENTS:-1 5 10 25 50}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

python src/data/make_data_scaling_datasets.py \
  --source-dir "${SOURCE_DIR}" \
  --output-root "${DATA_ROOT}"

python src/models/check_compute.py \
  --report-out "${OUTPUT_ROOT}/compute_check_gpu${GPU_ID}.json"

python src/models/run_phase2_data_scaling.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --percents ${PERCENTS}

python src/eval/summarize_scaling.py \
  --data-root "${OUTPUT_ROOT}" \
  --out "${PROJECT_ROOT}/outputs/phase2_scaling/tables/scaling_group_summary.csv"

python src/eval/path_length_analysis.py \
  --run "flan-t5-base_100" "${PROJECT_ROOT}/outputs/phase1_baselines_flan_bf16/flan-t5-base" \
  --run "flan-t5-base_multiscale" "${PROJECT_ROOT}/outputs/phase2_sft_method/multiscale/flan-t5-base" \
  --run "flan-t5-base_executor_feedback" "${PROJECT_ROOT}/outputs/phase2_sft_method/executor_feedback/flan-t5-base" \
  --run "flan-t5-base_dpo" "${PROJECT_ROOT}/outputs/phase2_executor_feedback_variants/dpo/flan-t5-base" \
  --run "flan-t5-base_grpo_rlvr" "${PROJECT_ROOT}/outputs/phase2_executor_feedback_variants/grpo_rlvr/flan-t5-base" \
  --out "${PROJECT_ROOT}/outputs/phase2_scaling/tables/path_length_summary.csv"

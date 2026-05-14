#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-auto}"
MIN_FREE_MB="${MIN_FREE_MB:-10240}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/data/generated/phase0/jsonl}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/generated/phase0}"
PREF_DIR="${PREF_DIR:-${PROJECT_ROOT}/data/generated/phase2_executor_preferences}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_executor_feedback_variants}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/outputs/phase1_baselines_flan_bf16/flan-t5-base/best}"
MODEL="${MODEL:-flan-t5-base}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
mkdir -p "${OUTPUT_ROOT}/logs"

select_gpu() {
  local line
  local index
  local used
  local total
  local util
  local free
  if [ "${GPU_ID}" != "auto" ]; then
    echo "${GPU_ID}"
    return
  fi
  while IFS=, read -r index used total util; do
    index=$(echo "${index}" | tr -d ' ')
    used=$(echo "${used}" | tr -d ' ')
    total=$(echo "${total}" | tr -d ' ')
    util=$(echo "${util}" | tr -d ' ')
    free=$((total - used))
    if [ "${free}" -ge "${MIN_FREE_MB}" ] && [ "${util}" -lt "${MAX_GPU_UTIL}" ]; then
      echo "${index}"
      return
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
  echo ""
}

wait_for_gpu() {
  local selected
  while true; do
    selected=$(select_gpu)
    if [ -n "${selected}" ]; then
      GPU_ID="${selected}"
      break
    fi
    echo "Waiting for any GPU with >=${MIN_FREE_MB}MiB free and <${MAX_GPU_UTIL}% util"
    sleep 120
  done
  echo "Selected GPU ${GPU_ID}"
}

write_dpo_config() {
  python - <<PY
import json
from pathlib import Path
cfg = {
    "model_name_or_path": "${BASE_CHECKPOINT}",
    "reference_model_name_or_path": "${BASE_CHECKPOINT}",
    "output_dir": "${OUTPUT_ROOT}/dpo/${MODEL}",
    "train_path": "${PREF_DIR}/train.jsonl",
    "eval_path": "${PREF_DIR}/val.jsonl",
    "beta": 0.1,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 1e-5,
    "eval_steps": 500,
    "early_stopping_patience": 5,
    "bf16": True
}
path = Path("${OUTPUT_ROOT}/dpo/${MODEL}/train_config.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2) + "\\n")
PY
}

write_grpo_config() {
  python - <<PY
import json
from pathlib import Path
cfg = {
    "model_name_or_path": "${BASE_CHECKPOINT}",
    "reference_model_name_or_path": "${BASE_CHECKPOINT}",
    "output_dir": "${OUTPUT_ROOT}/grpo_rlvr/${MODEL}",
    "train_path": "${SOURCE_DIR}/train.jsonl",
    "eval_path": "${SOURCE_DIR}/val.jsonl",
    "group_size": 4,
    "num_train_epochs": 3,
    "max_steps": 3000,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-6,
    "kl_coef": 0.02,
    "eval_steps": 250,
    "early_stopping_patience": 8,
    "max_eval_samples": 512,
    "bf16": True
}
path = Path("${OUTPUT_ROOT}/grpo_rlvr/${MODEL}/train_config.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2) + "\\n")
PY
}

wait_for_gpu
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [ ! -f "${PREF_DIR}/train.jsonl" ] || [ ! -f "${PREF_DIR}/val.jsonl" ]; then
  python src/data/make_executor_preference_data.py \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${PREF_DIR}" \
    --checkpoint "${BASE_CHECKPOINT}" \
    --batch-size 16 \
    --num-beams 8
else
  echo "Preference data already exists at ${PREF_DIR}; skipping generation."
fi

write_dpo_config
if [ ! -f "${OUTPUT_ROOT}/dpo/${MODEL}/train_summary.json" ] || [ ! -d "${OUTPUT_ROOT}/dpo/${MODEL}/best" ]; then
  python src/models/train_seq2seq_dpo.py --config "${OUTPUT_ROOT}/dpo/${MODEL}/train_config.json"
else
  echo "DPO checkpoint already exists; skipping DPO training."
fi

python src/models/run_phase1_baselines.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-root "${OUTPUT_ROOT}/dpo" \
  --models "${MODEL}" \
  --allow-cpu

write_grpo_config
if [ ! -f "${OUTPUT_ROOT}/grpo_rlvr/${MODEL}/train_summary.json" ] || [ ! -d "${OUTPUT_ROOT}/grpo_rlvr/${MODEL}/best" ]; then
  python src/models/train_seq2seq_grpo.py --config "${OUTPUT_ROOT}/grpo_rlvr/${MODEL}/train_config.json"
else
  echo "GRPO/RLVR checkpoint already exists; skipping GRPO/RLVR training."
fi

python src/models/run_phase1_baselines.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-root "${OUTPUT_ROOT}/grpo_rlvr" \
  --models "${MODEL}" \
  --allow-cpu

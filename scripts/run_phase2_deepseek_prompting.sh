#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-NLP}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/generated/phase0/jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_prompting_api/deepseek-v4-flash}"
MODEL="${MODEL:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DIRECT_MAX_NEW_TOKENS="${DIRECT_MAX_NEW_TOKENS:-256}"
REASONING_MAX_NEW_TOKENS="${REASONING_MAX_NEW_TOKENS:-1536}"

PROMPT_LIST="${PROMPT_LIST:-zero_shot one_shot few_shot cot react structured}"
SPLIT_LIST="${SPLIT_LIST:-test_unseen_placement test_unseen_environment ood_size_5x5 ood_size_7x7 ood_size_8x8 ood_size_9x9 ood_size_10x10 ood_dense_6x6 ood_dense_5x5 ood_dense_7x7 ood_aspect_10x5 ood_aspect_6x9}"
read -r -a PROMPTS <<< "${PROMPT_LIST}"
read -r -a SPLITS <<< "${SPLIT_LIST}"

cd "${PROJECT_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"
mkdir -p "${OUTPUT_ROOT}/logs"

if [ -z "${DS_API:-}" ]; then
  echo "DS_API is not set; export DS_API before running DeepSeek prompting." >&2
  exit 2
fi

# httpx in this environment does not have SOCKS support; keep HTTP(S) proxy
# variables if present, but avoid ALL_PROXY=socks5://... from interactive shells.
unset ALL_PROXY
unset all_proxy

for prompt in "${PROMPTS[@]}"; do
  max_new_tokens="${DIRECT_MAX_NEW_TOKENS}"
  thinking="disabled"
  if [ "${prompt}" = "cot" ]; then
    max_new_tokens="${REASONING_MAX_NEW_TOKENS}"
    thinking="enabled"
  fi
  for split in "${SPLITS[@]}"; do
    split_out="${OUTPUT_ROOT}/${prompt}/${split}"
    predictions="${split_out}/predictions.jsonl"
    metrics="${split_out}/metrics.json"
    if [ -f "${metrics}" ]; then
      echo "skip ${MODEL}/${prompt}/${split}: metrics already exists"
      continue
    fi
    mkdir -p "${split_out}"
    extra=()
    if [ -n "${MAX_SAMPLES}" ]; then
      extra+=(--max-samples "${MAX_SAMPLES}")
    fi
    python src/models/infer_deepseek_prompting.py \
      --model "${MODEL}" \
      --base-url "${BASE_URL}" \
      --prompt-type "${prompt}" \
      --dataset "${DATASET_DIR}/${split}.jsonl" \
      --out "${predictions}" \
      --max-new-tokens "${max_new_tokens}" \
      --thinking "${thinking}" \
      --concurrency "${CONCURRENCY}" \
      "${extra[@]}"
    python src/eval/evaluate_predictions.py \
      --predictions "${predictions}" \
      --dataset "${DATASET_DIR}/${split}.jsonl" \
      --metrics-out "${metrics}" \
      --predictions-out "${split_out}/evaluated_predictions.jsonl" \
      --failure-breakdown-out "${split_out}/failure_breakdown.json"
  done
done

python src/eval/summarize_prompting.py \
  --root "${PROJECT_ROOT}/outputs/phase2_prompting_api" \
  --models "${MODEL}" \
  --out "${OUTPUT_ROOT}/tables/prompting_group_summary.csv"

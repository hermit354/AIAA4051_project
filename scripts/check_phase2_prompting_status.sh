#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_prompting}"
MODEL_LIST="${MODEL_LIST:-t5-small t5-base flan-t5-small flan-t5-base bart-base}"
PROMPT_LIST="${PROMPT_LIST:-zero_shot one_shot few_shot cot react structured}"
read -r -a MODELS <<< "${MODEL_LIST}"
read -r -a PROMPTS <<< "${PROMPT_LIST}"

for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    dir="${OUTPUT_ROOT}/${model}/${prompt}"
    if [ ! -d "${dir}" ]; then
      echo "${model}/${prompt}: not started"
      continue
    fi
    count=$(find "${dir}" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l)
    echo "${model}/${prompt}: metrics ${count}/12"
  done
done

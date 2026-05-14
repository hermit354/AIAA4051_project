#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase2_input_format}"
MODEL="${MODEL:-flan-t5-base}"
FORMATS=("$@")

if [ "${#FORMATS[@]}" -eq 0 ]; then
  FORMATS=(structured grid_matrix hybrid)
fi

for format in "${FORMATS[@]}"; do
  dir="${OUTPUT_ROOT}/${MODEL}/${format}"
  if [ ! -d "${dir}" ]; then
    echo "${format}: not started"
    continue
  fi
  if [ -f "${dir}/train_summary.json" ]; then
    best=$(python -c "import json; print(json.load(open('${dir}/train_summary.json')).get('best_exact_match'))")
    echo "${format}: training done best_val_exact=${best}"
  else
    echo "${format}: training pending/running"
  fi
  count=$(find "${dir}" -name metrics.json | wc -l)
  echo "${format}: metrics ${count}/12"
done

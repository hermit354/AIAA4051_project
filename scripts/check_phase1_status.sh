#!/usr/bin/env bash
set -euo pipefail

# Experiment launcher: runs one reproducible training, evaluation, or status-check command.
# Configure PROJECT_ROOT, CONDA_BASE, ENV_NAME, GPU_ID, DATA_ROOT, or OUTPUT_ROOT as needed.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/phase1_baselines_converged}"
MODELS=(t5-small t5-base flan-t5-small flan-t5-base bart-base)
SPLITS=(
  test_unseen_placement
  test_unseen_environment
  ood_size_5x5
  ood_size_7x7
  ood_size_8x8
  ood_size_9x9
  ood_size_10x10
  ood_dense_6x6
  ood_dense_5x5
  ood_dense_7x7
  ood_aspect_10x5
  ood_aspect_6x9
)

cd "${PROJECT_ROOT}"

echo "Phase 1 status at $(date -u)"
echo "Output root: ${OUTPUT_ROOT}"
echo

for model in "${MODELS[@]}"; do
  model_dir="${OUTPUT_ROOT}/${model}"
  echo "== ${model} =="
  if [ -f "${model_dir}/train_summary.json" ]; then
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("train: done step={step} best_step={best_step} best_exact={best_exact_match:.4f}".format(**d))' "${model_dir}/train_summary.json"
  elif [ -f "${model_dir}/last_eval_metrics.json" ]; then
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("train: running/partial step={step} epoch={epoch} exact={exact_match:.4f} loss={loss:.4f}".format(**d))' "${model_dir}/last_eval_metrics.json"
  else
    echo "train: not started"
  fi

  done_splits=0
  for split in "${SPLITS[@]}"; do
    if [ -f "${model_dir}/${split}/metrics.json" ]; then
      done_splits=$((done_splits + 1))
    fi
  done
  echo "eval splits: ${done_splits}/${#SPLITS[@]}"
done

echo
if command -v tmux >/dev/null 2>&1; then
  tmux list-sessions 2>/dev/null || true
fi


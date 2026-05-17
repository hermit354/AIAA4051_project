#!/usr/bin/env bash
set -euo pipefail

# Download the full generated experiment inputs from Hugging Face and unpack
# them into the repository's ignored data/generated/ directory.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_DATASET_REPO="${HF_DATASET_REPO:-ouy-not-reversed/aiaa4051-path-planning-data}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$PROJECT_ROOT/data_external/aiaa4051-path-planning-data}"
TARGET_ROOT="${TARGET_ROOT:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"

if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "huggingface_hub is not installed for PYTHON_BIN=$PYTHON_BIN" >&2
  echo "Install it with: pip install huggingface_hub" >&2
  exit 1
fi

echo "Downloading dataset repo: $HF_DATASET_REPO"
echo "Local download directory: $DOWNLOAD_DIR"

HF_DATASET_REPO="$HF_DATASET_REPO" DOWNLOAD_DIR="$DOWNLOAD_DIR" "$PYTHON_BIN" -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    local_dir=os.environ["DOWNLOAD_DIR"],
    local_dir_use_symlinks=False,
)
'

echo "Extracting archives into: $TARGET_ROOT"
mkdir -p "$TARGET_ROOT"

shopt -s nullglob
archives=("$DOWNLOAD_DIR"/*.tar.gz)

if [ "${#archives[@]}" -eq 0 ]; then
  echo "No .tar.gz archives found in $DOWNLOAD_DIR" >&2
  exit 1
fi

for archive in "${archives[@]}"; do
  echo "Extracting $(basename "$archive")"
  tar -xzf "$archive" -C "$TARGET_ROOT"
done

echo "Done. Generated data should now be available under $TARGET_ROOT/data/generated/."

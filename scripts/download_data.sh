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

echo "Downloading dataset repo: $HF_DATASET_REPO"
echo "Local download directory: $DOWNLOAD_DIR"

HF_DATASET_REPO="$HF_DATASET_REPO" DOWNLOAD_DIR="$DOWNLOAD_DIR" "$PYTHON_BIN" - <<'PY'
import hashlib
import os
import urllib.request
from pathlib import Path

import json

repo_id = os.environ["HF_DATASET_REPO"]
download_dir = Path(os.environ["DOWNLOAD_DIR"])
download_dir.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import get_token
except Exception:
    get_token = lambda: None

token = get_token()
headers = {}
if token:
    headers["Authorization"] = f"Bearer {token}"


def download_file(filename: str) -> Path:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    out_path = download_dir / filename
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {filename}")
    with urllib.request.urlopen(request, timeout=120) as response, out_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return out_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = download_file("DATA_MANIFEST.json")
download_file("README.md")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

requested = os.environ.get("DOWNLOAD_PACKAGES")
requested_names = set(requested.split(",")) if requested else None

for package in manifest["packages"]:
    if requested_names and package["name"] not in requested_names and package["archive"] not in requested_names:
        continue
    archive_path = download_file(package["archive"])
    actual = sha256(archive_path)
    expected = package["sha256"]
    if actual != expected:
        raise SystemExit(f"Checksum mismatch for {package['archive']}: {actual} != {expected}")

if requested_names:
    known = {pkg["name"] for pkg in manifest["packages"]} | {pkg["archive"] for pkg in manifest["packages"]}
    missing = requested_names - known
    if missing:
        raise SystemExit(f"Unknown DOWNLOAD_PACKAGES entries: {', '.join(sorted(missing))}")

print(f"Downloaded files to {download_dir}")
PY

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

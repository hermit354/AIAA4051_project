"""Upload prepared dataset artifacts to a Hugging Face Dataset repository.

Run this after `scripts/prepare_hf_dataset.py` has created archives under
`hf_dataset_artifacts/` and after Hugging Face authentication is configured.

Typical caller:
    python scripts/upload_hf_dataset.py --repo-id ouy-not-reversed/aiaa4051-path-planning-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ouy-not-reversed/aiaa4051-path-planning-data")
    parser.add_argument("--artifact-dir", type=Path, default=Path("hf_dataset_artifacts"))
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Skip repository creation. Use this when the dataset repo was created in the browser.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload generated input data packages",
        help="Commit message for the Hugging Face dataset upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")

    files = sorted(path for path in artifact_dir.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in artifact directory: {artifact_dir}")

    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token found. Run scripts/check_hf_token.py --save first.")

    if not args.skip_create:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=token,
        )

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(artifact_dir),
        path_in_repo=".",
        commit_message=args.commit_message,
    )
    print(f"Uploaded {len(files)} files to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

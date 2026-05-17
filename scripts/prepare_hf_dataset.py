"""Prepare generated experiment input data for Hugging Face Dataset hosting.

This script packages the report-related generated input directories into
separate tar archives and writes a manifest with source paths, package sizes,
record counts, and SHA-256 checksums. It intentionally excludes raw upstream
data, checkpoints, logs, prediction dumps, and full outputs.

Typical caller:
    python scripts/prepare_hf_dataset.py --output-dir hf_dataset_artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DataPackage:
    """Description of one generated-data package to host externally."""

    name: str
    source: Path
    archive_name: str
    report_role: str


PACKAGES = [
    DataPackage(
        name="phase0",
        source=Path("data/generated/phase0"),
        archive_name="phase0.tar.gz",
        report_role="Base train, validation, ID, and OOD splits for Phase 0 and Phase 1.",
    ),
    DataPackage(
        name="phase2_input_format",
        source=Path("data/generated/phase2_input_format"),
        archive_name="phase2_input_format.tar.gz",
        report_role="Natural, structured, grid-matrix, and hybrid input-format diagnostics.",
    ),
    DataPackage(
        name="phase2_scaling",
        source=Path("data/generated/phase2_scaling"),
        archive_name="phase2_scaling.tar.gz",
        report_role="Data-scaling diagnostic datasets.",
    ),
    DataPackage(
        name="phase2_sft_method",
        source=Path("data/generated/phase2_sft_method"),
        archive_name="phase2_sft_method.tar.gz",
        report_role="SFT target-format and trace-supervision diagnostic datasets.",
    ),
    DataPackage(
        name="phase2_sizegen_trace",
        source=Path("data/generated/phase2_size_generalization/react_trace_compact_sizegen"),
        archive_name="phase2_sizegen_trace.tar.gz",
        report_role="Compact ReAct-style trace data for size-generalization experiments.",
    ),
    DataPackage(
        name="phase2_sizegen_heldout",
        source=Path("data/generated/phase2_size_generalization/react_trace_compact_sizegen_heldout"),
        archive_name="phase2_sizegen_heldout.tar.gz",
        report_role="Clean held-out large-size data for transfer evaluation.",
    ),
    DataPackage(
        name="phase2_dense_large",
        source=Path("data/generated/phase2_size_generalization/dense_large_small"),
        archive_name="phase2_dense_large.tar.gz",
        report_role="Dense large-grid stress-test data.",
    ),
    DataPackage(
        name="phase2_stepwise_policy",
        source=Path("data/generated/phase2_size_generalization/stepwise_policy"),
        archive_name="phase2_stepwise_policy.tar.gz",
        report_role="Plain stepwise-policy training and evaluation data.",
    ),
    DataPackage(
        name="phase2_stepwise_valid_actions",
        source=Path("data/generated/phase2_size_generalization/stepwise_valid_actions"),
        archive_name="phase2_stepwise_valid_actions.tar.gz",
        report_role="Stepwise data with valid-action signals.",
    ),
    DataPackage(
        name="phase2_stepwise_progress_target",
        source=Path("data/generated/phase2_size_generalization/stepwise_progress_target"),
        archive_name="phase2_stepwise_progress_target.tar.gz",
        report_role="Stepwise data with progress-target supervision.",
    ),
    DataPackage(
        name="phase2_stepwise_history",
        source=Path("data/generated/phase2_size_generalization/stepwise_progress_history_only"),
        archive_name="phase2_stepwise_history.tar.gz",
        report_role="Stepwise data with recent trajectory history in the input.",
    ),
    DataPackage(
        name="phase2_loop_dagger_history_topk",
        source=Path("data/generated/phase2_size_generalization/stepwise_loop_dagger_history_topk"),
        archive_name="phase2_loop_dagger_history_topk.tar.gz",
        report_role="Loop-targeted DAgger correction data for the final stepwise method.",
    ),
]


def iter_files(root: Path) -> Iterable[Path]:
    """Yield files under root in deterministic order."""

    return sorted(path for path in root.rglob("*") if path.is_file())


def bytes_to_mb(num_bytes: int) -> float:
    """Convert bytes to MiB for readable manifest values."""

    return round(num_bytes / 1024 / 1024, 3)


def count_jsonl_lines(files: Iterable[Path]) -> int:
    """Count JSONL records in the package without loading files into memory."""

    total = 0
    for path in files:
        if path.suffix == ".jsonl":
            with path.open("rb") as handle:
                total += sum(1 for _ in handle)
    return total


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for an archive using streaming reads."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_directory_to_tar(archive_path: Path, source: Path, project_root: Path) -> None:
    """Create a tar.gz archive that restores files at their repository path."""

    with tarfile.open(archive_path, mode="w:gz") as tar:
        tar.add(source, arcname=source.relative_to(project_root))


def build_manifest_entry(package: DataPackage, archive_path: Path, project_root: Path) -> dict:
    """Build a manifest entry after a package has been archived."""

    source = project_root / package.source
    files = list(iter_files(source))
    source_bytes = sum(path.stat().st_size for path in files)
    archive_bytes = archive_path.stat().st_size
    max_file = max(files, key=lambda path: path.stat().st_size) if files else None
    return {
        "name": package.name,
        "archive": archive_path.name,
        "source_path": str(package.source),
        "report_role": package.report_role,
        "source_size_mb": bytes_to_mb(source_bytes),
        "archive_size_mb": bytes_to_mb(archive_bytes),
        "file_count": len(files),
        "jsonl_file_count": sum(1 for path in files if path.suffix == ".jsonl"),
        "json_file_count": sum(1 for path in files if path.suffix == ".json"),
        "jsonl_record_count": count_jsonl_lines(files),
        "largest_source_file": str(max_file.relative_to(project_root)) if max_file else None,
        "largest_source_file_mb": bytes_to_mb(max_file.stat().st_size) if max_file else 0,
        "sha256": sha256_file(archive_path),
    }


def write_hf_readme(output_dir: Path, repo_id: str) -> None:
    """Write a concise Hugging Face Dataset card into the artifact directory."""

    readme = f"""---
license: other
task_categories:
- text-generation
language:
- en
pretty_name: AIAA4051 Grid Path Planning Generated Inputs
---

# AIAA4051 Grid Path Planning Generated Inputs

This dataset repository hosts generated experiment input data for the GitHub
project `{repo_id.replace('-data', '')}`. The GitHub repository contains code,
documentation, small samples, and lightweight result summaries. Full generated
inputs are hosted here because they are too large for regular GitHub commits.

The archives restore files under `data/generated/` when extracted at the root of
the GitHub repository.

Raw upstream data, model checkpoints, logs, prediction dumps, API keys, and full
outputs are intentionally excluded.

See `DATA_MANIFEST.json` for package sizes, checksums, source paths, and report
roles.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("hf_dataset_artifacts"))
    parser.add_argument("--repo-id", default="ouy-not-reversed/aiaa4051-path-planning-data")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional package names to build. Defaults to all report-related packages.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing archive instead of rebuilding it.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write README/manifest metadata for existing archives without creating archives.",
    )
    parser.add_argument(
        "--compression",
        choices=["gz"],
        default="gz",
        help="Archive compression format. Currently gz is used for portability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = PACKAGES
    if args.only:
        wanted = set(args.only)
        selected = [package for package in PACKAGES if package.name in wanted]
        missing = wanted - {package.name for package in selected}
        if missing:
            raise SystemExit(f"Unknown package name(s): {', '.join(sorted(missing))}")

    manifest = {
        "repo_id": args.repo_id,
        "created_by": "scripts/prepare_hf_dataset.py",
        "policy": "Generated inputs only. Raw upstream data, checkpoints, logs, predictions, and outputs are excluded.",
        "packages": [],
    }

    for package in selected:
        source = project_root / package.source
        if not source.exists():
            raise FileNotFoundError(f"Missing source directory: {source}")
        archive_path = output_dir / package.archive_name
        if not args.manifest_only and not (args.skip_existing and archive_path.exists()):
            print(f"Building {archive_path.name} from {package.source}")
            add_directory_to_tar(archive_path, source, project_root)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive does not exist: {archive_path}")
        manifest["packages"].append(build_manifest_entry(package, archive_path, project_root))

    write_hf_readme(output_dir, args.repo_id)
    manifest_path = output_dir / "DATA_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()

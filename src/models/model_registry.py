"""Map short experiment model names to local model directories.

Pipeline stage: shared model-resolution support for training and inference runners.
Inputs: user-facing model aliases such as t5-small or flan-t5-base.
Outputs: resolved paths under source_models and sorted model-name lists.
Typical caller: baseline, prompting, and SFT experiment runners.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODELS = ROOT / "source_models"

SEQ2SEQ_MODELS = {
    "t5-small": SOURCE_MODELS / "t5-small",
    "t5-base": SOURCE_MODELS / "t5-base",
    "flan-t5-small": SOURCE_MODELS / "flan-t5-small",
    "flan-t5-base": SOURCE_MODELS / "flan-t5-base",
    "bart-base": SOURCE_MODELS / "bart-base",
}


def model_path(name: str) -> Path:
    if name not in SEQ2SEQ_MODELS:
        known = ", ".join(sorted(SEQ2SEQ_MODELS))
        raise KeyError(f"unknown model {name!r}; known models: {known}")
    return SEQ2SEQ_MODELS[name]


def model_names() -> list[str]:
    return sorted(SEQ2SEQ_MODELS)

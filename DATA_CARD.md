# Data Card

This repository keeps only small examples in Git. Full generated experiment inputs are stored externally because the report-scale datasets are several gigabytes and include files larger than GitHub's regular file-size limits.

## Hosted Dataset

The intended Hugging Face Dataset repository is:

```text
ouy-not-reversed/aiaa4051-path-planning-data
```

After the dataset is uploaded, use:

```bash
bash scripts/download_data.sh
```

By default, this downloads the dataset repository into `data_external/aiaa4051-path-planning-data` and extracts the packaged generated data back into `data/generated/`.

## What Is Tracked in Git

| Path | Purpose | Git policy |
| --- | --- | --- |
| `data/samples/path_planning_sample.jsonl` | Minimal grid examples for evaluator smoke tests. | Tracked. |
| `data/samples/sample_predictions.jsonl` | Matching toy predictions for the sample dataset. | Tracked. |
| `data/raw/README.md` | Explains upstream raw data provenance. | Tracked. |
| `data/raw/ppnl_single_goal/` | Local upstream raw data. | Ignored. |
| `data/raw/ppnl_single_goal_original/` | Local upstream raw data in original layout. | Ignored. |
| `data/generated/**` | Full generated experiment inputs. | Ignored and hosted externally. |

## Report-Related Generated Inputs

These are the generated input datasets used by the final report experiments. Sizes are local measurements from the current project workspace.

| Package | Source path | Approx. size | Role in report |
| --- | --- | ---: | --- |
| `phase0.tar.gz` | `data/generated/phase0` | 229 MB | Base train, validation, ID, and OOD splits for Phase 0 and Phase 1. |
| `phase2_input_format.tar.gz` | `data/generated/phase2_input_format` | 366 MB | Natural, structured, grid-matrix, and hybrid input-format diagnostics. |
| `phase2_scaling.tar.gz` | `data/generated/phase2_scaling` | 429 MB | Data-scaling diagnostics. |
| `phase2_sft_method.tar.gz` | `data/generated/phase2_sft_method` | 596 MB | SFT target-format and trace-supervision diagnostics. |
| `phase2_sizegen_trace.tar.gz` | `data/generated/phase2_size_generalization/react_trace_compact_sizegen` | 221 MB | Compact trace size-generalization data. |
| `phase2_sizegen_heldout.tar.gz` | `data/generated/phase2_size_generalization/react_trace_compact_sizegen_heldout` | 221 MB | Clean held-out large-size evaluation data. |
| `phase2_dense_large.tar.gz` | `data/generated/phase2_size_generalization/dense_large_small` | 11 MB | Dense large-grid stress-test data. |
| `phase2_stepwise_policy.tar.gz` | `data/generated/phase2_size_generalization/stepwise_policy` | 624 MB | Plain stepwise-policy data. |
| `phase2_stepwise_valid_actions.tar.gz` | `data/generated/phase2_size_generalization/stepwise_valid_actions` | 683 MB | Stepwise data with valid-action signals. |
| `phase2_stepwise_progress_target.tar.gz` | `data/generated/phase2_size_generalization/stepwise_progress_target` | 659 MB | Stepwise data with progress targets. |
| `phase2_stepwise_history.tar.gz` | `data/generated/phase2_size_generalization/stepwise_progress_history_only` | 731 MB | Stepwise data with trajectory-history inputs. |
| `phase2_loop_dagger_history_topk.tar.gz` | `data/generated/phase2_size_generalization/stepwise_loop_dagger_history_topk` | 224 MB | Loop-targeted DAgger correction data. |

The full local `data/generated/` tree is about 7.9 GB. Several stepwise JSONL files exceed 100 MB, so committing them directly to GitHub is not appropriate.

## Raw Data Policy

Raw upstream PPNL-style files are not uploaded by default. They should only be redistributed if the course or upstream license explicitly allows it. The hosted dataset should contain generated inputs derived by this project, not private API keys, model checkpoints, logs, prediction dumps, or raw upstream archives.

## Rebuilding the Hugging Face Package Set

Prepare the external dataset artifacts locally:

```bash
python scripts/prepare_hf_dataset.py \
  --output-dir hf_dataset_artifacts \
  --compression gz
```

The command creates package archives and a `DATA_MANIFEST.json` with file sizes, SHA-256 checksums, source paths, and report roles.

## Uploading to Hugging Face

First authenticate locally. Create a Hugging Face token with write access at `https://huggingface.co/settings/tokens`, then run:

```bash
python -c "from huggingface_hub import login; login()"
```

After authentication, upload the prepared artifacts:

```bash
python scripts/upload_hf_dataset.py \
  --repo-id ouy-not-reversed/aiaa4051-path-planning-data
```

Use `--private` if the dataset should be private during review:

```bash
python scripts/upload_hf_dataset.py \
  --repo-id ouy-not-reversed/aiaa4051-path-planning-data \
  --private
```

## Expected Layout After Download

After running `scripts/download_data.sh`, the repository should contain:

```text
data/generated/phase0/
data/generated/phase2_input_format/
data/generated/phase2_scaling/
data/generated/phase2_sft_method/
data/generated/phase2_size_generalization/react_trace_compact_sizegen/
data/generated/phase2_size_generalization/react_trace_compact_sizegen_heldout/
data/generated/phase2_size_generalization/dense_large_small/
data/generated/phase2_size_generalization/stepwise_policy/
data/generated/phase2_size_generalization/stepwise_valid_actions/
data/generated/phase2_size_generalization/stepwise_progress_target/
data/generated/phase2_size_generalization/stepwise_progress_history_only/
data/generated/phase2_size_generalization/stepwise_loop_dagger_history_topk/
```

These paths match the experiment commands and report traceability notes.

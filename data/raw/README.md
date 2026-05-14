# Raw Data

This directory contains the PPNL single-goal benchmark files copied from:

`llms-as-path-planners/ppnl-spatial-temporal-reasoning`

The copied data is intentionally limited to the files this project needs. The full upstream repository is useful as a reference, but it is not required for running the training, inference, and evaluation pipeline.

## Directly Used Files

`ppnl_single_goal/` contains final single-goal samples with natural-language inputs and optimal action-sequence targets:

- `1_train_set_6x6_samples.json`
- `1dev_set_6x6_samples.json`
- `1_goals_test_seen_6x6_samples.json`
- `1goals_unseen_6x6_samples.json`
- `1_goals_test_unseen_5x5_samples.json`
- `1_goals_test_unseen_7x7_samples.json`
- `1_goals_test_unseen_6x6more_obstacles_samples.json`

These files can be converted into the generated JSON/JSONL datasets used by training, inference, and evaluation.

## Reference Files

`ppnl_single_goal_original/` contains the original split files from upstream `single_goal_original/`. They are kept for provenance and cross-checking, but the current training pipeline does not depend on them.

## Format

Each final sample includes:

- `world`: 2D grid where `0` is empty, `1` is obstacle, `2` is start, and `3` is goal.
- `nl_description`: model input text.
- `solution_coordinates`: optimal coordinate path.
- `agent_as_a_point`: target action sequence over `up`, `down`, `left`, `right`.
- `agent_has_direction`: oriented-agent commands, not used in this project.

Rows whose `agent_as_a_point` is `Goal not reachable` are skipped by the training loader and evaluator by default.

"""Generate data and path-statistics figures.

Pipeline stage: visualization after generated-data manifests are available.
Inputs: data manifests and path statistics from generated datasets.
Outputs: JSON, PNG, and SVG figure assets.
Typical caller: manual figure refresh for poster/report materials.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "data/generated/phase0/jsonl"
SIZEGEN = ROOT / "data/generated/phase2_size_generalization/react_trace_compact_sizegen/jsonl"
OUT_DIRS = [
    ROOT / "poster_assets",
    ROOT / "poster_claude_design_inputs",
]

SPLITS = [
    ("Train 6x6", PHASE0 / "train.jsonl"),
    ("Val 6x6", PHASE0 / "val.jsonl"),
    ("ID place", PHASE0 / "test_unseen_placement.jsonl"),
    ("ID env", PHASE0 / "test_unseen_environment.jsonl"),
    ("5x5", PHASE0 / "ood_size_5x5.jsonl"),
    ("7x7", PHASE0 / "ood_size_7x7.jsonl"),
    ("8x8", PHASE0 / "ood_size_8x8.jsonl"),
    ("9x9", PHASE0 / "ood_size_9x9.jsonl"),
    ("10x10", PHASE0 / "ood_size_10x10.jsonl"),
    ("Dense 5x5", PHASE0 / "ood_dense_5x5.jsonl"),
    ("Dense 6x6", PHASE0 / "ood_dense_6x6.jsonl"),
    ("Dense 7x7", PHASE0 / "ood_dense_7x7.jsonl"),
    ("10x5", PHASE0 / "ood_aspect_10x5.jsonl"),
    ("6x9", PHASE0 / "ood_aspect_6x9.jsonl"),
    ("12x12", SIZEGEN / "test_size_12x12.jsonl"),
    ("14x14", SIZEGEN / "test_size_14x14.jsonl"),
    ("16x16", SIZEGEN / "test_size_16x16.jsonl"),
]


def summarize(path: Path) -> dict[str, float]:
    count = 0
    total_shortest = 0.0
    total_manhattan = 0.0
    total_ratio = 0.0
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            shortest = float(ex["shortest_path_length"])
            manhattan = float(ex["manhattan_distance"])
            count += 1
            total_shortest += shortest
            total_manhattan += manhattan
            total_ratio += shortest / manhattan if manhattan else 1.0
    return {
        "examples": count,
        "avg_shortest_path": total_shortest / count,
        "avg_manhattan_distance": total_manhattan / count,
        "avg_shortest_over_manhattan": total_ratio / count,
    }


def main() -> None:
    rows = []
    for label, path in SPLITS:
        stats = summarize(path)
        rows.append({"split": label, **stats})

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "figure4_data_path_statistics.json").open("w") as f:
            json.dump(rows, f, indent=2)

    labels = [r["split"] for r in rows]
    examples = [r["examples"] for r in rows]
    shortest = [r["avg_shortest_path"] for r in rows]
    manhattan = [r["avg_manhattan_distance"] for r in rows]
    ratios = [r["avg_shortest_over_manhattan"] for r in rows]
    x = range(len(rows))

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(13.6, 6.6),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
        sharex=True,
        facecolor="white",
    )

    colors = ["#94a3b8"] * 4 + ["#60a5fa"] * 5 + ["#2dd4bf"] * 3 + ["#a78bfa"] * 2 + ["#fb923c"] * 3
    ax0.bar(x, examples, color=colors, edgecolor="white", linewidth=0.8)
    ax0.set_ylabel("Examples", fontsize=11)
    ax0.set_title("Dataset Coverage Across Training, ID, OOD, and Hard Size-Generalization Splits", fontsize=15, weight="bold")
    ax0.grid(axis="y", alpha=0.25)
    ax0.spines[["top", "right"]].set_visible(False)

    ax1.plot(x, shortest, marker="o", linewidth=2.3, color="#2563eb", label="Avg shortest path")
    ax1.plot(x, manhattan, marker="s", linewidth=2.0, color="#16a34a", label="Avg Manhattan distance")
    ax1b = ax1.twinx()
    ax1b.plot(x, ratios, marker="D", linewidth=1.9, color="#f97316", label="Shortest / Manhattan")
    ax1.set_ylabel("Average distance", fontsize=11)
    ax1b.set_ylabel("Ratio", fontsize=11, color="#9a3412")
    ax1b.tick_params(axis="y", labelcolor="#9a3412")
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1b.spines[["top"]].set_visible(False)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, loc="upper left", ncols=3, frameon=False)

    fig.text(
        0.012,
        0.01,
        "Path statistics are computed from the jsonl examples used by the project; ratio uses shortest_path_length / Manhattan_distance per example.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    for out_dir in OUT_DIRS:
        fig.savefig(out_dir / "figure4_data_path_statistics.svg")
        fig.savefig(out_dir / "figure4_data_path_statistics.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

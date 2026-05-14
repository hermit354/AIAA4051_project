"""Generate the main-results bar chart for the poster/report.

Pipeline stage: visualization after summary metrics are available.
Inputs: compact result values embedded in the script.
Outputs: PNG/SVG chart assets under poster asset directories.
Typical caller: manual figure refresh before report or poster submission.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "poster_assets", ROOT / "poster_claude_design_inputs"]

SPLITS = ["12x12", "14x14", "16x16"]
METHODS = [
    ("Full-path ReAct trace", [49.8, 29.0, 19.3], "#7E57C2"),
    ("Stepwise base", [64.9, 43.4, 30.8], "#2F6FDB"),
    ("Progress top-k", [88.5, 75.2, 62.8], "#009688"),
    ("Progress + lookahead", [94.5, 87.1, 77.2], "#2E7D32"),
    ("Loop-DAGGER + lookahead", [100.0, 100.0, 100.0], "#F28C28"),
]


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.8, 5.4), facecolor="white")
    x = list(range(len(SPLITS)))
    width = 0.15
    offsets = [(-2 + i) * width for i in range(len(METHODS))]

    for off, (name, vals, color) in zip(offsets, METHODS):
        xpos = [i + off for i in x]
        bars = ax.bar(xpos, vals, width, label=name, color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_title("Hard Size-Generalization: Stepwise Control Scales to Larger Grids", fontsize=15.5, weight="bold", pad=16)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_xticks(x, SPLITS, fontsize=12)
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncols=3,
        frameon=False,
        fontsize=9.5,
    )

    ax.annotate(
        "+57.9 pts on 16x16",
        xy=(2 + offsets[3], 77.2),
        xytext=(1.54, 55),
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.8),
        fontsize=12.5,
        color="#1B5E20",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F1F8E9", edgecolor="#B7D7A8"),
    )

    fig.text(
        0.012,
        0.012,
        "All methods are evaluated by simulator execution on 12x12, 14x14, and 16x16 hard size-generalization tests.",
        fontsize=9.2,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    for out_dir in OUT_DIRS:
        fig.savefig(out_dir / "figure2_baselines_best_method.png", dpi=240)
        fig.savefig(out_dir / "figure2_baselines_best_method.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

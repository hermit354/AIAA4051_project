"""Generate the full set of poster assets from curated results.

Pipeline stage: final visualization assembly.
Inputs: embedded summary values and selected example trajectories.
Outputs: PNG/SVG assets in poster_assets and design-input directories.
Typical caller: manual poster refresh before presentation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


OUT = Path("poster_assets")
OUT.mkdir(exist_ok=True)

COLORS = {
    "full": "#8A8F98",
    "react": "#7E57C2",
    "base": "#2F6FDB",
    "topk": "#009688",
    "lookahead": "#2E7D32",
    "dagger": "#F28C28",
    "red": "#C62828",
    "grid": "#D8DEE9",
    "obstacle": "#263238",
    "start": "#1976D2",
    "goal": "#2E7D32",
}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        if h != h:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 1.2,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main_results():
    labels = ["12x12", "14x14", "16x16"]
    methods = [
        ("Full-path ReAct trace", [49.8, 29.0, 19.3], COLORS["react"]),
        ("Stepwise base", [64.9, 43.4, 30.8], COLORS["base"]),
        ("Progress top-k", [88.5, 75.2, 62.8], COLORS["topk"]),
        ("Progress + lookahead", [94.5, 87.1, 77.2], COLORS["lookahead"]),
        ("Loop-DAGGER + lookahead", [100.0, 100.0, 100.0], COLORS["dagger"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(labels))
    width = 0.15
    offsets = [(-2 + i) * width for i in range(len(methods))]
    for off, (name, vals, color) in zip(offsets, methods):
        bars = ax.bar([i + off for i in x], vals, width, label=name, color=color)
        label_bars(ax, bars)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(list(x), labels)
    ax.set_title("Stepwise Control Scales to Larger Grids", fontweight="bold")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.legend(ncols=2, fontsize=8, frameon=False, loc="upper left")
    ax.text(
        2.1,
        57,
        "+57.9 pts\non 16x16",
        fontsize=13,
        fontweight="bold",
        color=COLORS["lookahead"],
        bbox=dict(boxstyle="round,pad=0.35", fc="#F1F8E9", ec="#B7D7A8"),
    )
    save(fig, "main_results_sizegen")


def phase1_baselines():
    labels = ["6x6 ID", "10x10 OOD", "Dense OOD", "Aspect OOD"]
    t5 = [97.5, 16.8, 74.5, 43.0]
    flan = [97.9, 17.7, 75.6, 44.2]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = range(len(labels))
    width = 0.34
    b1 = ax.bar([i - width / 2 for i in x], t5, width, label="T5-base", color="#AAB2BD")
    b2 = ax.bar([i + width / 2 for i in x], flan, width, label="Flan-T5-base", color=COLORS["full"])
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_title("Phase 1 Full-Path SFT Baselines", fontweight="bold")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        1.1,
        50,
        "High ID success,\nweak size transfer",
        fontsize=11,
        color=COLORS["red"],
        fontweight="bold",
    )
    save(fig, "phase1_baselines")


def ablation_16x16():
    labels = [
        "Stepwise\nbase",
        "Progress\ntop-k",
        "Progress +\nlookahead",
        "History-only\ntop-k",
        "History-only\nlookahead",
        "DAGGER\nno-history\n+ lookahead",
        "DAGGER\nhistory\n+ lookahead",
    ]
    vals = [30.8, 62.8, 77.2, 81.2, 90.5, 100.0, 100.0]
    colors = [
        COLORS["base"],
        COLORS["topk"],
        COLORS["lookahead"],
        "#4DB6AC",
        "#D9DDE3",
        "#FFB74D",
        COLORS["dagger"],
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    xs = range(len(labels))
    heights = [v if v is not None else 0 for v in vals]
    bars = ax.bar(xs, heights, color=colors)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(i, val + 1.4, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("16x16 success (%)")
    ax.set_xticks(list(xs), labels, fontsize=7.5)
    ax.set_title("What Drives the Best Stepwise Result?", fontweight="bold")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    save(fig, "stepwise_ablation_16x16")


def failure_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    types = ["Premature\nstop", "Hit\nobstacle", "Out of\nbounds", "Loop too\nlong"]
    counts = [12357, 2932, 1895, 72]
    bars = ax.bar(types, counts, color=[COLORS["red"], "#EF6C00", "#D84315", "#6A1B9A"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 350, f"{c:,}", ha="center", fontsize=8)
    ax.set_title("Full-path ReAct Failures", fontweight="bold")
    ax.set_ylabel("Count out of 46,794 examples")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)

    ax = axes[1]
    labels = ["Base stepwise\n16x16 failures", "Progress+lookahead\nresidual failures"]
    loop_rates = [98.6, 92.6]
    bars = ax.bar(labels, loop_rates, color=[COLORS["base"], COLORS["lookahead"]])
    for b, v in zip(bars, loop_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Loop-like among failures")
    ax.set_title("Residual Failure Mode", fontweight="bold")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    fig.suptitle("Failure Analysis: Corrections Target Loops and Backtracking", fontweight="bold")
    save(fig, "failure_analysis")


def draw_grid(ax, title, trajectory, color, annotate, obstacles, start, goal, shortest=None):
    n = 16
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=COLORS["grid"], lw=0.45))
    for r, c in obstacles:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=COLORS["obstacle"], ec=COLORS["obstacle"]))
    sr, sc = start
    gr, gc = goal
    ax.add_patch(Rectangle((sc - 0.5, sr - 0.5), 1, 1, fc=COLORS["start"], ec="white"))
    ax.text(sc, sr, "S", color="white", ha="center", va="center", fontweight="bold", fontsize=8)
    ax.add_patch(Rectangle((gc - 0.5, gr - 0.5), 1, 1, fc=COLORS["goal"], ec="white"))
    ax.text(gc, gr, "G", color="white", ha="center", va="center", fontweight="bold", fontsize=8)
    if trajectory:
        xs = [c for r, c in trajectory]
        ys = [r for r, c in trajectory]
        ax.plot(xs, ys, color=color, lw=2.1, alpha=0.9)
        for (r1, c1), (r2, c2) in zip(trajectory[:-1], trajectory[1:]):
            if (r1, c1) == (r2, c2):
                continue
            ax.add_patch(
                FancyArrowPatch(
                    (c1, r1),
                    (c2, r2),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    color=color,
                    lw=0.7,
                    alpha=0.9,
                )
            )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        -0.08,
        annotate,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        color="#333",
    )


def case_study():
    start = (0, 14)
    goal = (15, 1)
    obstacles = [(4, 13), (7, 3), (8, 0), (10, 3), (15, 13)]
    react_traj = [(0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (5, 13), (5, 12), (5, 11)]
    topk_traj = [
        (0, 14),
        (1, 14),
        (2, 14),
        (3, 14),
        (4, 14),
        (5, 14),
        (6, 14),
        (7, 14),
        (8, 14),
        (9, 14),
        (10, 14),
        (10, 13),
        (10, 12),
        (10, 11),
        (10, 10),
        (10, 9),
        (10, 8),
        (10, 7),
        (10, 6),
        (10, 5),
        (10, 4),
        (11, 4),
        (12, 4),
        (12, 3),
        (12, 2),
        (12, 1),
        (12, 0),
        (12, 1),
        (12, 0),
        (12, 1),
        (12, 0),
    ]
    final_traj = [
        (0, 14),
        (1, 14),
        (2, 14),
        (3, 14),
        (4, 14),
        (5, 14),
        (6, 14),
        (6, 13),
        (6, 12),
        (6, 11),
        (7, 11),
        (8, 11),
        (9, 11),
        (10, 11),
        (11, 11),
        (12, 11),
        (13, 11),
        (14, 11),
        (15, 11),
        (15, 10),
        (15, 9),
        (15, 8),
        (15, 7),
        (15, 6),
        (15, 5),
        (15, 4),
        (15, 3),
        (15, 2),
        (15, 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.2))
    draw_grid(
        axes[0],
        "Full-path ReAct",
        react_traj,
        COLORS["react"],
        "Premature stop after 8 moves\nfinal distance = 20",
        obstacles,
        start,
        goal,
    )
    draw_grid(
        axes[1],
        "Progress top-k",
        topk_traj,
        COLORS["topk"],
        "Near-goal oscillation\n(12,0) <-> (12,1)",
        obstacles,
        start,
        goal,
    )
    draw_grid(
        axes[2],
        "Loop-DAGGER + lookahead",
        final_traj,
        COLORS["dagger"],
        "Success + optimal\n28 steps",
        obstacles,
        start,
        goal,
    )
    fig.suptitle("Case Study: Same 16x16 Example, Different Failure Modes", fontweight="bold")
    save(fig, "case_study_16x16")


def workflow():
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    ax.axis("off")
    boxes = [
        ("Full-path\ntraining data", "grid, S, G,\nobstacles -> path"),
        ("State-level\nexpansion", "16,032 paths\n-> 92,455 states"),
        ("Stepwise\npolicy", "predict one\naction at a time"),
        ("Simulator +\nverifier", "mask invalid moves\n+ lookahead"),
    ]
    xs = [0.05, 0.31, 0.57, 0.83]
    for i, ((title, body), x) in enumerate(zip(boxes, xs)):
        ax.add_patch(Rectangle((x - 0.11, 0.28), 0.21, 0.44, fc="#F7F9FC", ec="#9AA7B7", lw=1.2))
        ax.text(x - 0.095, 0.61, title, fontsize=11, fontweight="bold", va="top")
        ax.text(x - 0.095, 0.47, body, fontsize=9, va="top")
        if i < len(xs) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.11, 0.5), (xs[i + 1] - 0.12, 0.5), arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#546A7B"))
    ax.text(
        0.5,
        0.08,
        r"$a_t = \arg\max$ verifier(top-k LM actions | state$_t$);   state$_{t+1}$ = executor(state$_t$, $a_t$)",
        ha="center",
        fontsize=10,
    )
    save(fig, "workflow_stepwise_executor")


def main():
    main_results()
    phase1_baselines()
    ablation_16x16()
    failure_analysis()
    case_study()
    workflow()


if __name__ == "__main__":
    main()

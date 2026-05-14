"""Generate the best-method workflow and result figure.

Pipeline stage: visualization for the final method explanation.
Inputs: selected metrics and illustrative grid/path values embedded in the script.
Outputs: PNG/SVG assets for the poster/report.
Typical caller: manual figure refresh before final submission.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster_claude_design_inputs"
OUT.mkdir(exist_ok=True)

# The poster currently uses a manually revised PNG:
# poster_claude_design_inputs/figure3_best_method.png
# Keep this script as a reproducible draft generator only, so running it will
# not overwrite the approved poster asset.

COLORS = {
    "data": "#4b5b73",
    "policy": "#1559d1",
    "policy_fill": "#f3f7ff",
    "teal": "#0f9488",
    "teal_fill": "#f0fdfa",
    "green": "#147d2c",
    "green_fill": "#f2fbf3",
    "orange": "#f97316",
    "orange_fill": "#fff7ed",
    "red": "#dc2626",
    "red_fill": "#fff1f2",
    "grid": "#cbd5e1",
    "obstacle": "#263241",
    "text": "#111827",
}


def box(ax, x, y, w, h, title, edge, fill="white", lw=1.6, title_color=None, title_size=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        weight="bold",
        color=title_color or edge,
    )
    return patch


def arrow(ax, start, end, color="#334155", lw=2.0, rad=0.0, style="-|>", ms=16, ls="-", alpha=1.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=ls,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def draw_grid(ax, x, y, size, n=6, start=(5, 0), goal=(0, 5), obstacles=(), path=(), path_color="#f97316"):
    cell = size / n
    ax.add_patch(Rectangle((x, y), size, size, facecolor="white", edgecolor="#94a3b8", linewidth=0.9))
    for i in range(n + 1):
        ax.plot([x, x + size], [y + i * cell, y + i * cell], color=COLORS["grid"], lw=0.55)
        ax.plot([x + i * cell, x + i * cell], [y, y + size], color=COLORS["grid"], lw=0.55)
    for r, c in obstacles:
        ax.add_patch(Rectangle((x + c * cell, y + (n - 1 - r) * cell), cell, cell, color=COLORS["obstacle"]))
    for label, rc, color in [("S", start, "#16a34a"), ("G", goal, "#2563eb")]:
        r, c = rc
        ax.add_patch(Rectangle((x + c * cell, y + (n - 1 - r) * cell), cell, cell, color=color))
        ax.text(
            x + (c + 0.5) * cell,
            y + (n - 1 - r + 0.5) * cell,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold",
        )
    if path:
        pts = [(x + (c + 0.5) * cell, y + (n - 1 - r + 0.5) * cell) for r, c in path]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=path_color, lw=2.2, marker="o", markersize=3.6, markerfacecolor=path_color)


def chip(ax, x, y, w, h, text, edge, fill="white", fontsize=8.5, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            edgecolor=edge,
            facecolor=fill,
            linewidth=1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["text"], weight=weight)


def draw_tree(ax, x, y, w, h):
    root = (x + w / 2, y + h * 0.78)
    mids = [(x + w * 0.25, y + h * 0.48), (x + w * 0.50, y + h * 0.48), (x + w * 0.75, y + h * 0.48)]
    leaves = [
        (x + w * 0.16, y + h * 0.20),
        (x + w * 0.30, y + h * 0.20),
        (x + w * 0.44, y + h * 0.20),
        (x + w * 0.56, y + h * 0.20),
        (x + w * 0.70, y + h * 0.20),
        (x + w * 0.84, y + h * 0.20),
    ]
    for p in mids:
        ax.plot([root[0], p[0]], [root[1], p[1]], color="#334155", lw=0.9)
    for i, p in enumerate(mids):
        for leaf in leaves[2 * i : 2 * i + 2]:
            ax.plot([p[0], leaf[0]], [p[1], leaf[1]], color="#334155", lw=0.8)
    for p in [root] + mids + leaves:
        ax.add_patch(Circle(p, 0.012, facecolor="#14b8a6", edgecolor="#0f766e", linewidth=0.8))


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 7.0), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "Loop-DAGGER History + Lookahead Decoding",
        ha="center",
        va="center",
        fontsize=19,
        weight="bold",
        color=COLORS["text"],
    )

    # Stage 1: data
    box(ax, 0.03, 0.30, 0.14, 0.50, "1. Stepwise data", COLORS["data"], "#f8fafc", title_size=10.5)
    oracle_path = [(5, 0), (5, 1), (5, 2), (4, 2), (3, 2), (3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5)]
    draw_grid(
        ax,
        0.045,
        0.50,
        0.11,
        start=(5, 0),
        goal=(0, 5),
        obstacles=[(1, 2), (3, 1), (4, 4)],
        path=oracle_path,
        path_color=COLORS["orange"],
    )
    chip(ax, 0.045, 0.40, 0.11, 0.055, "16,032 paths", COLORS["data"], "white", fontsize=7.8)
    chip(ax, 0.045, 0.335, 0.11, 0.055, "92,455 state-action\nexamples", COLORS["data"], "white", fontsize=7.0)

    # Stage 2: policy
    box(ax, 0.205, 0.34, 0.14, 0.46, "2. History-aware\nFlan-T5 policy", COLORS["policy"], COLORS["policy_fill"], title_size=9.8)
    for i, label in enumerate(["state", "valid actions", "history", "progress"]):
        chip(ax, 0.22, 0.665 - i * 0.068, 0.11, 0.047, label, COLORS["policy"], "white", fontsize=7.8)
    arrow(ax, (0.275, 0.39), (0.275, 0.355), color=COLORS["policy"], lw=1.5, ms=10)
    chip(ax, 0.22, 0.295, 0.11, 0.052, "top-k actions", COLORS["policy"], "white", fontsize=8.2, weight="bold")

    # Stage 3: rollout
    box(ax, 0.38, 0.34, 0.32, 0.46, "3. Simulator-guided top-k lookahead", COLORS["teal"], COLORS["teal_fill"], title_size=10.8)
    chip(ax, 0.405, 0.59, 0.075, 0.12, "propose\nactions", COLORS["teal"], "white", fontsize=7.6)
    for dx, dy, txt in [(0.027, 0.092, "↑"), (0.012, 0.058, "←"), (0.043, 0.058, "→"), (0.027, 0.024, "↓")]:
        ax.text(0.405 + dx, 0.59 + dy, txt, ha="center", va="center", fontsize=13, color=COLORS["teal"], weight="bold")
    sim = FancyBboxPatch((0.505, 0.59), 0.105, 0.12, boxstyle="round,pad=0.006,rounding_size=0.01", edgecolor=COLORS["teal"], facecolor="white", linewidth=1.0)
    ax.add_patch(sim)
    ax.text(0.557, 0.704, "simulate 2-3 steps", ha="center", va="center", fontsize=7.0, color=COLORS["text"])
    draw_tree(ax, 0.515, 0.592, 0.085, 0.100)
    chip(ax, 0.635, 0.59, 0.045, 0.12, "choose\naction\n↑", COLORS["teal"], "white", fontsize=7.2, weight="bold")
    arrow(ax, (0.482, 0.65), (0.505, 0.65), color=COLORS["teal"], lw=2.0)
    arrow(ax, (0.612, 0.65), (0.635, 0.65), color=COLORS["teal"], lw=2.0)
    arrow(ax, (0.658, 0.59), (0.435, 0.575), color=COLORS["teal"], lw=1.5, rad=-0.32, ms=12)
    chip(ax, 0.455, 0.515, 0.16, 0.040, "invalid moves masked", COLORS["teal"], "white", fontsize=7.4)
    chip(ax, 0.425, 0.445, 0.21, 0.040, "score: likelihood + distance + loop penalty", COLORS["teal"], "white", fontsize=7.1)

    # Stage 4: residual failure mining
    box(ax, 0.32, 0.08, 0.26, 0.18, "4. Mine residual loop failures", COLORS["red"], COLORS["red_fill"], title_size=9.6)
    loop_path = [(4, 0), (4, 1), (4, 2), (4, 1), (4, 2), (4, 3)]
    draw_grid(ax, 0.34, 0.115, 0.075, n=5, start=(4, 0), goal=(0, 4), obstacles=[(0, 4), (4, 4)], path=loop_path, path_color=COLORS["red"])
    arrow(ax, (0.425, 0.155), (0.465, 0.155), color=COLORS["red"], lw=2.0)
    chip(ax, 0.465, 0.12, 0.052, 0.07, "collect\nfailed\nstates", COLORS["red"], "white", fontsize=6.6)
    arrow(ax, (0.52, 0.155), (0.55, 0.155), color=COLORS["red"], lw=2.0)
    chip(ax, 0.552, 0.12, 0.022, 0.07, "★", COLORS["red"], "white", fontsize=14)
    ax.text(0.563, 0.105, "oracle\nactions", ha="center", va="top", fontsize=6.3, color=COLORS["text"])

    # Stage 5: correction
    box(ax, 0.62, 0.08, 0.18, 0.18, "5. Loop-DAGGER correction", COLORS["orange"], COLORS["orange_fill"], title_size=9.4)
    chip(ax, 0.64, 0.17, 0.135, 0.045, "failed states", COLORS["orange"], "white", fontsize=7.7)
    chip(ax, 0.64, 0.105, 0.135, 0.045, "oracle next actions", COLORS["orange"], "white", fontsize=7.7)

    # Final method
    box(ax, 0.855, 0.20, 0.125, 0.60, "6. Final method", COLORS["green"], COLORS["green_fill"], title_size=11.2)
    final_path = [(5, 0), (5, 1), (5, 2), (4, 2), (4, 3), (3, 3), (3, 4), (2, 4), (2, 5), (1, 5), (0, 5)]
    draw_grid(
        ax,
        0.875,
        0.50,
        0.09,
        start=(5, 0),
        goal=(0, 5),
        obstacles=[(2, 2), (4, 1), (3, 4)],
        path=final_path,
        path_color=COLORS["green"],
    )
    ax.text(0.917, 0.42, "history-aware policy", ha="center", va="center", fontsize=8.2, color=COLORS["green"], weight="bold")
    ax.text(0.917, 0.36, "+ Loop-DAGGER correction", ha="center", va="center", fontsize=8.0, color=COLORS["green"], weight="bold")
    ax.text(0.917, 0.30, "+ top-k lookahead decoding", ha="center", va="center", fontsize=8.0, color=COLORS["green"], weight="bold")
    ax.plot([0.885, 0.955], [0.255, 0.255], color=COLORS["green"], lw=1.0)
    ax.text(0.917, 0.225, "executable, loop-free,\ngoal-reaching path", ha="center", va="top", fontsize=7.1, color=COLORS["text"])

    # Main arrows and feedback.
    arrow(ax, (0.17, 0.55), (0.205, 0.55), color="#64748b", lw=2.3, ms=17)
    arrow(ax, (0.345, 0.55), (0.38, 0.55), color=COLORS["policy"], lw=2.3, ms=17)
    arrow(ax, (0.70, 0.57), (0.855, 0.57), color=COLORS["green"], lw=2.5, ms=18)
    arrow(ax, (0.52, 0.34), (0.46, 0.26), color=COLORS["red"], lw=1.6, ms=12)
    arrow(ax, (0.58, 0.17), (0.62, 0.17), color=COLORS["red"], lw=2.0, ms=14)
    arrow(ax, (0.80, 0.17), (0.855, 0.34), color=COLORS["orange"], lw=1.8, ms=13)
    arrow(ax, (0.62, 0.26), (0.275, 0.345), color=COLORS["orange"], lw=1.2, rad=0.22, ms=11, ls="--", alpha=0.9)
    ax.text(0.45, 0.315, "fine-tune on loop corrections", ha="center", va="center", fontsize=6.8, color=COLORS["orange"], weight="bold")

    fig.savefig(OUT / "figure3_best_method_generated_draft.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "figure3_best_method_generated_draft.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

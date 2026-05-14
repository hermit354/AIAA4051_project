"""Generate an inset explaining simulator lookahead behavior.

Pipeline stage: visualization for method explanation.
Inputs: illustrative grid and candidate-path values embedded in the script.
Outputs: PNG/SVG inset assets for posters or slides.
Typical caller: manual figure refresh before presentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parents[1] / "poster_claude_design_inputs"


def draw_grid(ax, x0, y0, size=2.25, label="", agent=None, path=None, candidate=False):
    n = 4
    cell = size / n
    ax.text(x0 + size / 2, y0 + size + 0.18, label, ha="center", va="bottom",
            fontsize=16, fontweight="bold", color="#111111")

    # Cells: row 0 is top, col 0 is left.
    start = (3, 0)
    goal = (0, 3)
    obstacle = (2, 1)

    for r in range(n):
        for c in range(n):
            x = x0 + c * cell
            y = y0 + (n - 1 - r) * cell
            face = "white"
            edge = "#9aa3ad"
            if (r, c) == obstacle:
                face = "#1f2933"
            elif (r, c) == start:
                face = "#1557d8"
            elif (r, c) == goal:
                face = "#0b7a1e"
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=face,
                                   edgecolor=edge, linewidth=1.1))

    def center(rc):
        r, c = rc
        return x0 + (c + 0.5) * cell, y0 + (n - 1 - r + 0.5) * cell

    if candidate:
        sx, sy = center(start)
        ex, ey = center((2, 0))
        ax.add_patch(FancyArrowPatch((sx, sy + 0.22), (ex, ey - 0.12),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.6, linestyle="--",
                                     color="#0b7a1e"))

    if path:
        pts = [center(p) for p in path]
        scatter_pts = pts
        if path[0] == start and len(path) > 1:
            sx, sy = center(start)
            # Begin rollout at the upper edge of S so the S label stays readable.
            pts[0] = (sx, sy + cell * 0.42)
            scatter_pts = pts[1:]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#0b7a1e", linewidth=3,
                solid_capstyle="round", zorder=5)
        if scatter_pts:
            sxs, sys = zip(*scatter_pts)
            ax.scatter(sxs, sys, s=95, color="#0b7a1e", edgecolor="white",
                       linewidth=1.2, zorder=6)

    if agent and agent != start:
        ax.scatter(*center(agent), s=140, color="#0b7a1e",
                   edgecolor="white", linewidth=1.6, zorder=7)

    # Draw cell letters last so path dots never hide S/G.
    ax.text(*center(start), "S", ha="center", va="center",
            fontsize=18, color="white", fontweight="bold", zorder=9)
    ax.text(*center(goal), "G", ha="center", va="center",
            fontsize=18, color="white", fontweight="bold", zorder=9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=220)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    border = FancyBboxPatch(
        (0.15, 0.15), 13.7, 4.3,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=2.2, linestyle=(0, (5, 5)),
        edgecolor="#0b6b1b", facecolor="white"
    )
    ax.add_patch(border)
    ax.text(7, 4.16, "simulate 2-3 steps ahead", ha="center", va="center",
            fontsize=24, fontweight="bold", color="#0b6b1b")

    y0 = 0.92
    x_positions = [0.9, 5.85, 10.8]
    draw_grid(ax, x_positions[0], y0, label="current state",
              candidate=True)
    draw_grid(ax, x_positions[1], y0, label="step +1",
              path=[(3, 0), (2, 0)], agent=(2, 0))
    # Valid 3-step rollout: up, up, right. It never enters obstacle (2, 1).
    draw_grid(ax, x_positions[2], y0, label="step +2/3",
              path=[(3, 0), (2, 0), (1, 0), (1, 1)],
              agent=(1, 1))

    def connect(x1, x2, text):
        y = 2.35
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=1.7,
                                     color="#0b7a1e"))
        ax.text((x1 + x2) / 2, y + 0.22, text, ha="center", va="bottom",
                fontsize=10.5, color="#0b6b1b", fontweight="bold",
                linespacing=1.0)

    connect(3.45, 5.25, "roll out\ncandidate")
    connect(8.4, 10.2, "score\nfuture state")

    ax.text(7, 0.42,
            "choose next action by simulated progress, valid moves, and loop penalty",
            ha="center", va="center", fontsize=13, color="#111111")

    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"figure3_simulate_lookahead_inset.{ext}",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

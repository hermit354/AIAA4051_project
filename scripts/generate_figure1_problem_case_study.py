"""Generate the problem case-study figure for the poster/report.

Pipeline stage: visualization after representative examples have been selected.
Inputs: hard-coded case-study layout values in this script.
Outputs: PNG/SVG figure assets under poster asset directories.
Typical caller: manual figure refresh before report or poster submission.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "poster_assets",
    ROOT / "poster_claude_design_inputs",
]

CASE_ID = "test_size_16x16_16x16_obs5_sizegen_test_size_16x16031_pair008"
ROWS = 16
COLS = 16
START = (0, 14)
GOAL = (15, 1)
OBSTACLES = [(4, 13), (7, 3), (8, 0), (10, 3), (15, 13)]
FAILED_TRACE = [
    (0, 14),
    (1, 14),
    (2, 14),
    (3, 14),
    (4, 14),
    (5, 14),
    (5, 13),
    (5, 12),
    (5, 11),
]
FAILED_ACTIONS = "down down down down down left left left"
SHORTEST_PATH_LENGTH = 28
FINAL_DISTANCE_TO_GOAL = 20


def cell_center(rc):
    r, c = rc
    return c + 0.5, r + 0.5


def draw_grid(ax):
    ax.set_xlim(0, COLS)
    ax.set_ylim(ROWS, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(COLS + 1))
    ax.set_yticks(range(ROWS + 1))
    ax.grid(color="#d7dee8", linewidth=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#8a94a6")
        spine.set_linewidth(1.1)

    for r, c in OBSTACLES:
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#2f3744", edgecolor="#1f2630", linewidth=0.8))

    sr, sc = START
    gr, gc = GOAL
    ax.add_patch(Rectangle((sc, sr), 1, 1, facecolor="#2563eb", edgecolor="#1d4ed8", linewidth=1.0))
    ax.add_patch(Rectangle((gc, gr), 1, 1, facecolor="#16a34a", edgecolor="#15803d", linewidth=1.0))
    ax.text(sc + 0.5, sr + 0.5, "S", ha="center", va="center", color="white", weight="bold", fontsize=12)
    ax.text(gc + 0.5, gr + 0.5, "G", ha="center", va="center", color="white", weight="bold", fontsize=12)


def main():
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.0, 4.7), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 0.92], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    info = fig.add_subplot(gs[0, 1])

    draw_grid(ax)

    xs, ys = zip(*(cell_center(rc) for rc in FAILED_TRACE))
    ax.plot(xs, ys, color="#f97316", linewidth=4.2, solid_capstyle="round", zorder=5)
    ax.scatter(xs[1:-1], ys[1:-1], s=32, color="#fb923c", edgecolor="white", linewidth=0.7, zorder=6)
    ax.scatter([xs[-1]], [ys[-1]], s=120, marker="X", color="#dc2626", edgecolor="white", linewidth=1.0, zorder=7)

    gx, gy = cell_center(GOAL)
    ax.plot([xs[-1], gx], [ys[-1], gy], color="#dc2626", linewidth=1.7, linestyle=(0, (3, 3)), alpha=0.72)
    ax.annotate(
        "premature stop\n20 cells from G",
        xy=(xs[-1], ys[-1]),
        xytext=(9.2, 8.2),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        ha="left",
        va="center",
        fontsize=10.5,
        color="#991b1b",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2", edgecolor="#fecdd3"),
    )
    ax.set_title("Executable path failure on a 16x16 grid", fontsize=14, weight="bold", pad=8)

    info.axis("off")
    info.text(0.0, 0.98, "Real evaluated example", fontsize=13, weight="bold", color="#111827", va="top")
    info.text(0.0, 0.90, "ID: ...031_pair008", fontsize=9.5, color="#4b5563", va="top")
    info.text(0.0, 0.78, "Input", fontsize=11.5, weight="bold", color="#111827", va="top")
    info.text(0.0, 0.70, "grid size: 16x16\nstart: (0,14)\ngoal: (15,1)\nobstacles: 5", fontsize=10, color="#374151", linespacing=1.35, va="top")
    info.text(0.0, 0.45, "Full-path ReAct output", fontsize=11.5, weight="bold", color="#111827", va="top")
    info.text(0.0, 0.37, FAILED_ACTIONS, fontsize=10, color="#9a3412", wrap=True, va="top")
    info.text(0.0, 0.20, "Failure under simulator", fontsize=11.5, weight="bold", color="#991b1b", va="top")
    info.text(
        0.0,
        0.12,
        f"premature stop\nshortest path length: {SHORTEST_PATH_LENGTH}\nfinal distance-to-goal: {FINAL_DISTANCE_TO_GOAL} cells",
        fontsize=10,
        color="#374151",
        linespacing=1.25,
        va="top",
    )

    fig.suptitle("A fluent text path is not necessarily an executable plan", fontsize=15.5, weight="bold", y=0.985)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.87, bottom=0.055)

    for out_dir in OUT_DIRS:
        fig.savefig(out_dir / "figure1_problem_case_study.svg")
        fig.savefig(out_dir / "figure1_problem_case_study.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()

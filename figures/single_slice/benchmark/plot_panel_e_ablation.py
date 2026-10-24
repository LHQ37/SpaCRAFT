"""Draw Figure 2e from the frozen formal single-slice ablation tables."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
GRAPH_DATA = REPO / "results/ablation/single_slice_graph_ablation_mean.csv"
LOSS_DATA = (
    REPO / "results/ablation/single_slice_loss_ablation/"
    "single_slice_loss_ablation_mean.csv"
)
METRICS = [
    "ARI", "NMI", "AMI", "FMI", "Homogeneity", "Completeness",
    "V-measure",
]
DISPLAY_LABELS = {
    "Homogeneity": "Homo.",
    "Completeness": "Comp.",
    "V-measure": "V-meas.",
}
GRAPH_METHODS = ["Spatial Only", "Feature Only", "Hybrid Graph"]
LOSS_METHODS = ["Full", "w/o Reconstruction", "w/o InfoNCE", "w/o Adversarial"]
COLORS = {
    "Spatial Only": "#009E73", "Feature Only": "#E69F00",
    "Hybrid Graph": "#0072B2", "Full": "#0072B2",
    "w/o Reconstruction": "#56B4E9", "w/o InfoNCE": "#E69F00",
    "w/o Adversarial": "#D55E00",
}


def read_table(path, methods):
    frame = pd.read_csv(path).set_index("variant")
    if list(frame.loc[methods].index) != methods or not set(METRICS).issubset(frame):
        raise ValueError(f"invalid frozen single-slice ablation table: {path}")
    values = frame.loc[methods, METRICS].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"non-finite ablation value: {path}")
    return values


def _draw_grouped_bars(ax, table, methods, lower=0.0):
    x = np.arange(len(METRICS), dtype=float)
    width = 0.24 if len(methods) == 3 else 0.19
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for offset, method in zip(offsets, methods):
        values = table.loc[method].to_numpy()
        ax.bar(x + offset, values - lower, bottom=lower, width=width * 0.92,
               color=COLORS[method], edgecolor="#4A4A4A", linewidth=0.4,
               label=method, zorder=3)
    ax.set_xticks(x, [DISPLAY_LABELS.get(metric, metric) for metric in METRICS])
    ax.tick_params(axis="x", labelrotation=25, labelsize=6.4, length=0, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_ylabel("Score", fontsize=7.2)
    ax.tick_params(axis="y", labelsize=6.4, length=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E7E7E7", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")


def _draw_header(ax, title, methods):
    """Keep each title and legend inside its own dedicated header area."""
    ax.set_axis_off()
    ax.text(0.5, 0.78, title, ha="center", va="center", fontsize=8.2)
    ax.legend(handles=[Patch(facecolor=COLORS[name], edgecolor="#4A4A4A",
                             linewidth=0.4, label=name) for name in methods],
              loc="center", bbox_to_anchor=(0.5, 0.24), ncol=len(methods),
              frameon=False, fontsize=6.5, columnspacing=0.9,
              handlelength=1.1, handletextpad=0.35, borderaxespad=0.0)


def draw_panel(fig, spec):
    """Embed both approved single-slice ablations as one Figure 2e panel."""
    graph = read_table(GRAPH_DATA, GRAPH_METHODS)
    loss = read_table(LOSS_DATA, LOSS_METHODS)
    grid = spec.subgridspec(
        2, 2, height_ratios=[0.19, 0.81], hspace=0.015, wspace=0.22
    )
    graph_header = fig.add_subplot(grid[0, 0])
    loss_header = fig.add_subplot(grid[0, 1])
    graph_ax = fig.add_subplot(grid[1, 0])
    loss_ax = fig.add_subplot(grid[1, 1])

    _draw_header(graph_header, "Single-slice graph ablation", GRAPH_METHODS)
    _draw_header(loss_header,
                 "Single-slice loss ablation (6-dataset average)",
                 LOSS_METHODS)

    _draw_grouped_bars(graph_ax, graph, GRAPH_METHODS)
    graph_ax.set_ylim(0.0, 1.0)
    graph_ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    _draw_grouped_bars(loss_ax, loss, LOSS_METHODS, lower=0.4)
    loss_ax.set_ylim(0.4, 0.7)
    loss_ax.set_yticks([0.4, 0.5, 0.6, 0.7])

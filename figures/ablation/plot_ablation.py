#!/usr/bin/env python3
"""Generate the formal two-panel SpaCRAFT ablation figure.

Panel a reads the graph-ablation means recovered numerically from the existing
formal Matplotlib SVG (the original approved CSV was deleted with ``combine``).
Panel b reads the formal four-variant, six-dataset loss-ablation mean table.
No experimental value is calculated or changed by this plotting script.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GRAPH_DATA = REPO / "results/ablation/single_slice_graph_ablation_mean.csv"
LOSS_DATA = (
    REPO / "results/ablation/single_slice_loss_ablation/"
    "single_slice_loss_ablation_mean.csv"
)
OUT = Path(__file__).resolve().parent

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
LOSS_METHODS = [
    "Full", "w/o Reconstruction", "w/o InfoNCE", "w/o Adversarial",
]
COLORS = {
    "Spatial Only": "#009E73",
    "Feature Only": "#E69F00",
    "Hybrid Graph": "#0072B2",
    "Full": "#0072B2",
    "w/o Reconstruction": "#56B4E9",
    "w/o InfoNCE": "#E69F00",
    "w/o Adversarial": "#D55E00",
}
EDGE = "#4A4A4A"
GRID = "#E7E7E7"
FONT = (
    "Arial" if any(item.name == "Arial" for item in font_manager.fontManager.ttflist)
    else "Liberation Sans"
)

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 8.4,
    "axes.titlesize": 10.2,
    "axes.labelsize": 8.7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def read_table(path, methods):
    frame = pd.read_csv(path).set_index("variant")
    assert list(frame.loc[methods].index) == methods
    assert set(METRICS).issubset(frame.columns)
    values = frame.loc[methods, METRICS].astype(float)
    assert np.isfinite(values.to_numpy()).all()
    assert values.to_numpy().min() >= 0 and values.to_numpy().max() <= 1
    return values


def style_axis(ax, ylim, ticks):
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.set_ylabel("Score")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.65)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2.5, color="#777777")


def grouped_bars(ax, table, methods, *, lower=0.0):
    x = np.arange(len(METRICS), dtype=float)
    width = 0.24 if len(methods) == 3 else 0.19
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for offset, method in zip(offsets, methods):
        values = table.loc[method].to_numpy()
        ax.bar(
            x + offset,
            values - lower,
            bottom=lower,
            width=width * 0.92,
            color=COLORS[method],
            edgecolor=EDGE,
            linewidth=0.45,
            label=method,
            zorder=3,
        )
    ax.set_xticks(x, [DISPLAY_LABELS.get(metric, metric) for metric in METRICS])
    ax.tick_params(axis="x", labelrotation=27)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def add_legend(ax, methods):
    ax.legend(
        handles=[
            Patch(facecolor=COLORS[name], edgecolor=EDGE, linewidth=0.45, label=name)
            for name in methods
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.105),
        ncol=len(methods),
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.4,
    )


def main():
    graph = read_table(GRAPH_DATA, GRAPH_METHODS)
    loss = read_table(LOSS_DATA, LOSS_METHODS)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.78, wspace=0.24)

    grouped_bars(ax_a, graph, GRAPH_METHODS)
    style_axis(ax_a, (0.0, 1.0), np.arange(0.0, 1.01, 0.2))
    ax_a.set_title("Single-slice graph ablation", pad=8)
    add_legend(ax_a, GRAPH_METHODS)

    grouped_bars(ax_b, loss, LOSS_METHODS, lower=0.4)
    style_axis(ax_b, (0.4, 0.7), [0.4, 0.5, 0.6, 0.7])
    ax_b.set_title("Single-slice loss ablation (6-dataset average)", pad=8)
    add_legend(ax_b, LOSS_METHODS)

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.11, 1.08, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="left", va="bottom",
                clip_on=False)

    png = OUT / "Figure_ablation.png"
    pdf = OUT / "Figure_ablation.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"FINAL_PNG={png}")
    print(f"FINAL_PDF={pdf}")


if __name__ == "__main__":
    main()

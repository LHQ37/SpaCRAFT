#!/usr/bin/env python3
"""Restore the final legacy visual appearance of Figure 2 panel b.

The geometry is reconstructed from the final reference screenshot on a
1064 x 422 design grid.  All numerical values come from the frozen formal
``performance_summary_input.csv`` table; the screenshot is never used as data.
Rounded-square positions are visual reconstruction only because no surviving
source assigns biological or statistical meaning to marker shape.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUT = HERE / "performance_summary_input.csv"
OUTPUT = REPO / "figures/main"

METHODS = [
    "SpaCRAFT", "INSTINCT", "SnapATAC2", "Descart",
    "epiScanpy", "PeakVI", "SMOPCA", "SCALE",
]
DATASETS = [
    "s1_adata_atac_with_meta", "E11_0-S1_atac", "E13_5-S1_atac",
    "E15_5-S1_atac", "E18_5-S1_atac", "adata_peaks_normalized",
]
DATASET_LABELS = ["P22", "E11", "E13", "E15", "E18", "Emb"]
RANKING_METRICS = [
    "ARI", "NMI", "AMI", "FMI", "Homogeneity", "Completeness",
    "V-measure", "Overall",
]
RANKING_LABELS = [
    "ARI", "NMI", "AMI", "FMI", "Homo.", "Comp.", "V-meas.", "Overall",
]
# Keep the right-hand dataset summaries bit-for-bit tied to the original panel:
# these markers were defined as the mean of the four formerly displayed fields.
METRIC_AVERAGE_METRICS = ["NMI", "ARI", "AMI", "Overall"]

# Visual reconstruction from the final figure reference.  There is no surviving
# code or metadata indicating that these shapes encode a data variable.
ROUNDED_SQUARES = {
    ("SpaCRAFT", "P22"), ("SpaCRAFT", "E11"), ("SpaCRAFT", "Emb"),
    ("INSTINCT", "E18"),
    ("SnapATAC2", "P22"), ("SnapATAC2", "E13"),
    ("SMOPCA", "E13"), ("SMOPCA", "E15"),
    ("SCALE", "E15"),
}

RANK_CMAP = LinearSegmentedColormap.from_list(
    "legacy_rank", ["#E4E0F0", "#8177B5", "#35166F"]
)
# The final reference used the legacy YlOrBr-like direction: low values are
# deeper orange and high values are pale yellow.  Size still increases with the
# unchanged formal performance value.
METRIC_CMAP = LinearSegmentedColormap.from_list(
    "legacy_metric",
    [
        (0.00, "#D6450A"),
        (0.10, "#E95B0C"),
        (0.20, "#F27B17"),
        (0.30, "#FAA42E"),
        (0.40, "#FFC858"),
        (0.50, "#FFE391"),
        (0.60, "#FFF4BB"),
        (1.00, "#FFF9CC"),
    ],
)


def load_values():
    raw = pd.read_csv(INPUT)
    raw = raw[
        raw["method"].isin(METHODS) & raw["dataset"].isin(DATASETS)
    ].copy()
    if len(raw) != 48 or raw.duplicated(["method", "dataset"]).any():
        raise ValueError("formal benchmark source must contain exactly 8 x 6 rows")
    if not np.isfinite(raw[RANKING_METRICS].to_numpy()).all():
        raise ValueError("formal benchmark source contains non-finite values")

    ranked_blocks = []
    for dataset in DATASETS:
        block = raw[raw["dataset"] == dataset].copy()
        for metric in RANKING_METRICS:
            block[f"RANK_{metric}"] = block[metric].rank(
                ascending=False, method="min"
            )
        ranked_blocks.append(block)
    ranked = pd.concat(ranked_blocks, ignore_index=True)
    ranks = ranked.groupby("method")[
        [f"RANK_{metric}" for metric in RANKING_METRICS]
    ].mean().loc[METHODS]
    averages = (
        raw.assign(metric_average=raw[METRIC_AVERAGE_METRICS].mean(axis=1))
        .pivot(index="method", columns="dataset", values="metric_average")
        .loc[METHODS, DATASETS]
    )
    return ranks, averages


def add_rounded_square(ax, x, y, diameter, color, linewidth=0.75):
    ax.add_patch(FancyBboxPatch(
        (x - diameter / 2, y - diameter / 2), diameter, diameter,
        boxstyle=f"round,pad=0,rounding_size={max(2.5, diameter * 0.11)}",
        facecolor=color, edgecolor="#242424", linewidth=linewidth, zorder=4,
    ))


def draw_panel(fig=None, spec=None):
    ranks, averages = load_values()
    embedded = fig is not None
    text_scale = 0.68 if embedded else 1.0
    if fig is None:
        fig = plt.figure(figsize=(10.40, 4.36), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        if spec is None:
            raise ValueError("spec is required when drawing into an existing figure")
        ax = fig.add_subplot(spec)
    ax.set_xlim(0, 1040)
    ax.set_ylim(436, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    content_left, content_right = 82, 1037
    header_bottom, subheader_bottom = 37, 82
    row_edges = np.asarray([82, 125, 168, 211, 254, 297, 340, 383, 426])
    row_centers = (row_edges[:-1] + row_edges[1:]) / 2

    # Continuous legacy header band and alternating rows across all sections.
    ax.add_patch(Rectangle(
        (content_left, 0), content_right - content_left, header_bottom,
        facecolor="#DDDDDD", edgecolor="none", zorder=0,
    ))
    for row in range(8):
        if row % 2 == 0:
            ax.add_patch(Rectangle(
                (content_left, row_edges[row]),
                content_right - content_left, row_edges[row + 1] - row_edges[row],
                facecolor="#DEDEDE", edgecolor="none", zorder=0,
            ))

    # Panel and group headings.
    ax.text(23, 15, "b", fontsize=39 * text_scale, fontweight="bold", va="top", ha="center")
    ax.text(145, 19, "Method", fontsize=11.2 * text_scale, fontweight="bold",
            ha="center", va="center")
    ax.text(480, 19, "Ranking", fontsize=11.2 * text_scale, fontweight="bold",
            ha="center", va="center")
    ax.text(894, 19, "Metric averages by dataset", fontsize=11.2 * text_scale,
            fontweight="bold", ha="center", va="center")

    # Give long method names (notably SnapATAC2) a comfortable gutter before
    # the ranking matrix while retaining the dataset-average section width.
    rank_left, rank_right = 215.0, 750.0
    rank_cell_width = (rank_right - rank_left) / len(RANKING_METRICS)
    rank_baselines = rank_left + np.arange(len(RANKING_METRICS)) * rank_cell_width
    rank_right_edges = rank_baselines + rank_cell_width - 4.0
    metric_centers = np.asarray([790, 833, 877, 920, 963, 1007], dtype=float)
    rank_label_centers = rank_left + (np.arange(len(RANKING_METRICS)) + 0.5) * rank_cell_width
    for x, label in zip(rank_label_centers, RANKING_LABELS):
        ax.text(x, 59, label, fontsize=8.4 * text_scale, fontweight="bold",
                ha="center", va="center")
    for x, label in zip(metric_centers, DATASET_LABELS):
        ax.text(x, 59, label, fontsize=10.3 * text_scale, fontweight="bold",
                ha="center", va="center")

    # Strong dashed rank baselines and section separator.
    for baseline in rank_baselines:
        ax.plot([baseline, baseline], [83, 426], color="#111111",
                linewidth=1.25, linestyle=(0, (7, 6)), zorder=5)
    ax.plot([750, 750], [83, 426], color="#111111",
            linewidth=1.25, linestyle=(0, (7, 6)), zorder=5)

    score_norm = Normalize(0, 1)
    metric_norm = Normalize(0, 1)
    for row, method in enumerate(METHODS):
        y = row_centers[row]
        ax.text(145, y, method, fontsize=10.4 * text_scale,
                fontweight="bold" if method == "SpaCRAFT" else "semibold",
                ha="center", va="center", color="#202020")

        for column, metric in enumerate(RANKING_METRICS):
            rank = float(ranks.loc[method, f"RANK_{metric}"])
            rank_score = (8.0 - rank) / 7.0
            available = rank_right_edges[column] - rank_baselines[column]
            width = available * rank_score
            # SAGE score 0 is represented by a true zero-length bar.
            if width > 0:
                ax.add_patch(Rectangle(
                    (rank_baselines[column], y - 18.5), width, 37,
                    facecolor=RANK_CMAP(score_norm(rank_score)),
                    edgecolor="#202020", linewidth=0.75, zorder=3,
                ))

        for column, dataset_label in enumerate(DATASET_LABELS):
            value = float(averages.iloc[row, column])
            # Monotone soft-threshold mapping: weak values remain dot-like,
            # medium values separate gradually, and strong values approach the
            # restrained legacy maximum diameter.  No value is altered.
            diameter = 3.2 + 35.5 / (1.0 + np.exp(-(value - 0.42) / 0.065))
            color = METRIC_CMAP(metric_norm(value))
            x = metric_centers[column]
            if (method, dataset_label) in ROUNDED_SQUARES:
                add_rounded_square(ax, x, y, diameter, color)
            else:
                ax.add_patch(Circle(
                    (x, y), diameter / 2, facecolor=color,
                    edgecolor="#242424", linewidth=0.75, zorder=4,
                ))

    # Compact legends reconstructed from the reference placement.
    ax.text(13, 342, "Rank-based score", fontsize=5.2 * text_scale,
            ha="left", va="bottom")
    legend_x0, legend_y, legend_width = 13, 357, 62
    for index in range(32):
        ax.add_patch(Rectangle(
            (legend_x0 + legend_width * index / 32, legend_y),
            legend_width / 32 + 0.2, 12,
            facecolor=RANK_CMAP(index / 31), edgecolor="none", zorder=2,
        ))
    ax.text(13, 375, "0", fontsize=5.1 * text_scale, ha="left")
    ax.text(75, 375, "1", fontsize=5.1 * text_scale, ha="right")

    ax.text(13, 394, "overall", fontsize=5.2 * text_scale, ha="left", va="bottom")
    legend_values = np.linspace(0, 1, 7)
    legend_centers = [16, 24, 32, 41, 51, 63, 77]
    for index, value in enumerate(legend_values):
        # Increasing gaps prevent the larger high-value symbols from touching.
        x = legend_centers[index]
        # Reuse the main-panel size progression at a compact legend scale.
        diameter = 2.5 + 9.5 / (1.0 + np.exp(-(value - 0.42) / 0.09))
        if index >= 5:
            add_rounded_square(
                ax, x, 410, diameter, METRIC_CMAP(value), linewidth=0.65
            )
        else:
            ax.add_patch(Circle(
                (x, 410), diameter / 2, facecolor=METRIC_CMAP(value),
                edgecolor="#444444", linewidth=0.45, zorder=4,
            ))
    ax.text(13, 428, "0", fontsize=5.1 * text_scale, ha="left")
    ax.text(82, 428, "1", fontsize=5.1 * text_scale, ha="right")
    return fig, ranks, averages


def main():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, _, _ = draw_panel()
    pdf = OUTPUT / "panel_b_sage_rank_score.pdf"
    png = OUTPUT / "panel_b_sage_rank_score.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=400, facecolor="white")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()

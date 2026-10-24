#!/usr/bin/env python3
"""Build the complete, vector-first Figure 2 from frozen formal inputs."""

from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

from domain_palette import spatial_domain_colors
from plot_panel_b_restored import draw_panel as draw_refined_panel_b
from plot_panel_e_ablation import draw_panel as draw_ablation_panel


REPO = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO / "figures/main"
DATA = REPO / "data/E13_5-S1_atac.h5ad"
SUMMARY = BENCHMARK_DIR / "performance_summary_input.csv"
CLUSTER_ROOT = REPO / "results/single_slice/benchmark/oracle_cluster_labels"
UMAP_ROOT = REPO / "results/single_slice/benchmark/oracle_umap_coordinates"

METHODS = [
    "SpaCRAFT", "INSTINCT", "Descart", "SnapATAC2",
    "epiScanpy", "PeakVI", "SMOPCA", "SCALE",
]
PANEL_B_METHODS = [
    "SpaCRAFT", "INSTINCT", "SnapATAC2", "Descart",
    "epiScanpy", "PeakVI", "SMOPCA", "SCALE",
]
DATASETS = [
    "s1_adata_atac_with_meta", "E11_0-S1_atac", "E13_5-S1_atac",
    "E15_5-S1_atac", "E18_5-S1_atac", "adata_peaks_normalized",
]
DATASET_LABELS = ["P22", "E11", "E13", "E15", "E18", "Emb"]
METRICS = [
    "ARI", "NMI", "AMI", "FMI", "Homogeneity", "Completeness",
    "V-measure", "Overall",
]
METRIC_AVERAGE_METRICS = ["NMI", "ARI", "AMI", "Overall"]


def aligned_column(path, obs_names, column):
    frame = pd.read_csv(path, dtype={"obs_name": str})
    required = {"obs_name", column}
    if not required.issubset(frame.columns) or frame["obs_name"].duplicated().any():
        raise ValueError(f"invalid frozen input: {path}")
    values = frame.set_index("obs_name")[column].reindex(obs_names)
    if values.isna().any():
        raise ValueError(f"{path} is missing {int(values.isna().sum())} spots")
    return values.to_numpy()


def aligned_umap(path, obs_names):
    frame = pd.read_csv(path, dtype={"obs_name": str})
    required = {"obs_name", "UMAP1", "UMAP2"}
    if not required.issubset(frame.columns) or frame["obs_name"].duplicated().any():
        raise ValueError(f"invalid frozen input: {path}")
    values = frame.set_index("obs_name")[["UMAP1", "UMAP2"]].reindex(obs_names)
    if values.isna().any().any():
        raise ValueError(f"{path} has missing coordinates")
    return values.to_numpy()


def match_clusters(predicted, truth):
    predicted = np.asarray(predicted).astype(str)
    truth = np.asarray(truth).astype(str)
    pred_levels = pd.Categorical(predicted).categories.tolist()
    true_levels = pd.Categorical(truth).categories.tolist()
    table = np.zeros((len(pred_levels), len(true_levels)), dtype=int)
    pred_index = {value: i for i, value in enumerate(pred_levels)}
    true_index = {value: i for i, value in enumerate(true_levels)}
    for pred, true in zip(predicted, truth):
        table[pred_index[pred], true_index[true]] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {pred_levels[row]: true_levels[col] for row, col in zip(rows, cols)}
    for row, value in enumerate(pred_levels):
        mapping.setdefault(value, true_levels[int(np.argmax(table[row]))])
    return np.asarray([mapping[value] for value in predicted])


def benchmark_tables():
    raw = pd.read_csv(SUMMARY)
    expected = set(METHODS) | set(raw["method"])
    if len(raw) != 48 or expected != set(METHODS):
        raise ValueError("formal benchmark summary does not contain the expected 8 x 6 rows")
    raw = raw[
        raw["method"].isin(METHODS) & raw["dataset"].isin(DATASETS)
    ].copy()
    if raw.duplicated(["method", "dataset"]).any():
        raise ValueError("duplicate method/dataset benchmark rows")

    rank_rows = []
    for dataset in DATASETS:
        block = raw[raw["dataset"] == dataset].copy()
        for metric in METRICS:
            block[f"RANK_{metric}"] = block[metric].rank(
                ascending=False, method="min"
            )
        rank_rows.append(block)
    ranked = pd.concat(rank_rows)
    ranks = ranked.groupby("method")[
        [f"RANK_{metric}" for metric in METRICS]
    ].mean()
    # Figure 2b uses the fixed manuscript display order.  This is deliberately
    # independent of the numerical ranks and does not imply a ranking.
    order = PANEL_B_METHODS
    averages = (
        raw.assign(metric_average=raw[METRIC_AVERAGE_METRICS].mean(axis=1))
        .pivot(index="method", columns="dataset", values="metric_average")
        .loc[order, DATASETS]
    )
    return raw, ranks.loc[order], averages, order


def clean_spatial_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_panel_label(fig, spec, label):
    box = spec.get_position(fig)
    fig.text(box.x0 - 0.018, box.y1 - 0.002, label,
             fontsize=14, fontweight="bold", ha="left", va="top")


def plot_panel_a(fig, spec, coordinates, truth, colors):
    inner = spec.subgridspec(1, 2, width_ratios=[1.78, 0.92], wspace=0.005)
    ax = fig.add_subplot(inner[0, 0])
    ax.scatter(coordinates[:, 0], coordinates[:, 1],
               c=[colors[value] for value in truth], s=5.2,
               linewidths=0, alpha=1.0, rasterized=False)
    clean_spatial_axis(ax)
    ax.set_title("E13.5 ground truth", fontsize=9, pad=2)

    legend_ax = fig.add_subplot(inner[0, 1])
    legend_ax.axis("off")
    handles = [
        Line2D([], [], linestyle="", marker="o", markersize=4.6,
               markerfacecolor=colour, markeredgecolor="none",
               label=label.replace("_", " "))
        for label, colour in colors.items()
    ]
    legend_ax.legend(handles=handles, loc="center left", frameon=False,
                     fontsize=5.6, handletextpad=0.35, labelspacing=0.42,
                     borderaxespad=0.0)


def plot_panel_b(fig, spec):
    _, ranks, averages, order = benchmark_tables()
    inner = spec.subgridspec(
        3, 3, height_ratios=[0.15, 0.70, 0.15],
        width_ratios=[1.65, 3.35, 3.85], hspace=0.02, wspace=0.035,
    )
    headers = ["Method", "Ranking score", "Metric averages by dataset"]
    for column, title in enumerate(headers):
        ax = fig.add_subplot(inner[0, column])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(Rectangle(
            (0, 0.10), 1, 0.78, transform=ax.transAxes,
            facecolor="#EEEEEE", edgecolor="none", zorder=0,
        ))
        ax.text(0.5, 0.52, title, ha="center", va="center",
                fontsize=8.2, fontweight="bold")

    method_ax = fig.add_subplot(inner[1, 0])
    rank_ax = fig.add_subplot(inner[1, 1])
    metric_ax = fig.add_subplot(inner[1, 2])
    n_methods = len(order)
    for ax in (method_ax, rank_ax, metric_ax):
        ax.set_ylim(n_methods - 0.5, -0.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    method_ax.set_xlim(0, 1)
    method_ax.set_xticks([])
    method_ax.set_yticks([])
    for row, method in enumerate(order):
        if row % 2 == 0:
            method_ax.axhspan(row - 0.5, row + 0.5, color="#F2F2F2", zorder=0)
        method_ax.text(
            0.04, row, method, va="center", ha="left", fontsize=7.2,
            color="#333333",
            fontweight="bold" if method == "SpaCRAFT" else "normal",
        )

    rank_values = ranks.to_numpy(dtype=float)
    rank_cmap = LinearSegmentedColormap.from_list(
        "figure2_rank", ["#F2EEF8", "#B6A3D1", "#5B2A86"]
    )
    rank_ax.set_xlim(0, len(METRICS))
    rank_ax.set_xticks(np.arange(len(METRICS)) + 0.5, METRICS,
                       fontsize=6.4, fontweight="bold")
    rank_ax.xaxis.tick_top()
    rank_ax.set_yticks([])
    for row in range(n_methods):
        if row % 2 == 0:
            rank_ax.axhspan(row - 0.5, row + 0.5, color="#F2F2F2", zorder=0)
    baseline_offset = 0.05
    maximum_bar_width = 0.88
    for column in range(len(METRICS)):
        rank_ax.axvline(
            column + baseline_offset, color="#333333", linewidth=0.65,
            linestyle=(0, (3, 2)), zorder=3,
        )
    for row in range(n_methods):
        for column in range(len(METRICS)):
            raw_rank = rank_values[row, column]
            ranking_score = 1.0 - (raw_rank - 1.0) / 7.0
            width = maximum_bar_width * ranking_score
            rank_ax.add_patch(Rectangle(
                (column + baseline_offset, row - 0.39), max(width, 0.012), 0.78,
                facecolor=rank_cmap(ranking_score),
                edgecolor="#2F2F2F", linewidth=0.8, zorder=2,
            ))

    metric_values = averages.to_numpy(dtype=float)
    metric_ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    metric_ax.set_xticks(range(len(DATASETS)), DATASET_LABELS,
                         fontsize=6.4, fontweight="bold")
    metric_ax.xaxis.tick_top()
    metric_ax.set_yticks([])
    for row in range(n_methods):
        if row % 2 == 0:
            metric_ax.axhspan(row - 0.5, row + 0.5, color="#F2F2F2", zorder=0)
    cmap = LinearSegmentedColormap.from_list(
        "figure2_performance", ["#FFF7C2", "#FFC857", "#F28E2B"]
    )
    norm = Normalize(0, 1)
    for row in range(n_methods):
        for column in range(len(DATASETS)):
            value = metric_values[row, column]
            metric_ax.scatter(column, row, s=18 + 250 * value,
                              c=[cmap(norm(value))], edgecolors="#333333",
                              linewidths=0.75, zorder=2)

    metric_ax.plot(
        [-0.48, -0.48], [-0.48, n_methods - 0.52],
        color="#333333", linewidth=0.7, linestyle=(0, (3, 2)),
        clip_on=False,
    )

    blank_legend = fig.add_subplot(inner[2, 0])
    blank_legend.axis("off")

    rank_legend = fig.add_subplot(inner[2, 1])
    rank_legend.set_xlim(0, 1)
    rank_legend.set_ylim(0, 1)
    rank_legend.axis("off")
    rank_legend.text(0.03, 0.82, "Ranking score", fontsize=6.2,
                     fontweight="bold", ha="left", va="center")
    segments = 24
    for index in range(segments):
        rank_legend.add_patch(Rectangle(
            (0.03 + 0.52 * index / segments, 0.28),
            0.52 / segments, 0.22,
            facecolor=rank_cmap(index / (segments - 1)), edgecolor="none",
        ))
    rank_legend.text(0.03, 0.10, "0  worse", fontsize=5.5, ha="left")
    rank_legend.text(0.55, 0.10, "1  better", fontsize=5.5, ha="right")
    rank_legend.text(0.75, 0.40, "higher = better", fontsize=5.7,
                     fontweight="bold", ha="center", va="center")

    metric_legend = fig.add_subplot(inner[2, 2])
    metric_legend.set_xlim(0, 1)
    metric_legend.set_ylim(0, 1)
    metric_legend.axis("off")
    metric_legend.text(0.03, 0.82, "Overall", fontsize=6.2,
                       fontweight="bold", ha="left", va="center")
    legend_values = [0.0, 0.5, 1.0]
    legend_x = [0.09, 0.25, 0.46]
    for x, value in zip(legend_x, legend_values):
        metric_legend.scatter(
            x, 0.36, s=18 + 250 * value, c=[cmap(norm(value))],
            edgecolors="#333333", linewidths=0.75,
        )
        metric_legend.text(x, 0.07, f"{value:g}", fontsize=5.5,
                           ha="center", va="center")


def add_umap_axes(ax):
    """Restore the UMAP coordinate arrows used in the original Figure 2d."""
    # Both arrows share one exact origin.  Disabling endpoint shrink removes
    # the small visual gap that otherwise appears between their line caps.
    origin = (0.020, 0.008)
    ax.annotate(
        "", xy=(0.305, origin[1]), xytext=origin, xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "#222222", "lw": 1.15,
                    "mutation_scale": 10, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )
    ax.annotate(
        "", xy=(origin[0], 0.308), xytext=origin, xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "#222222", "lw": 1.15,
                    "mutation_scale": 10, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )
    ax.text(0.165, -0.060, "UMAP1", transform=ax.transAxes,
            fontsize=6.2, ha="center", va="bottom", color="#222222")
    ax.text(-0.030, 0.158, "UMAP2", transform=ax.transAxes,
            fontsize=6.2, ha="left", va="center", rotation=90,
            color="#222222")


def save_panel_b():
    """Save the standalone Figure 2b component from the formal frozen table."""
    configure_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.47, 3.45), facecolor="white")
    spec = fig.add_gridspec(1, 1, left=0.045, right=0.99,
                            bottom=0.08, top=0.97)[0]
    plot_panel_b(fig, spec)
    add_panel_label(fig, spec, "b")
    pdf = OUTPUT_DIR / "panel_b.pdf"
    png = OUTPUT_DIR / "panel_b.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=350, facecolor="white")
    plt.close(fig)
    print(pdf)
    print(png)


def plot_comparison_grid(fig, spec, coordinates_by_method, point_colors,
                         spatial=False):
    grid = spec.subgridspec(2, 4, wspace=0.055, hspace=0.08)
    for index, method in enumerate(METHODS):
        ax = fig.add_subplot(grid[index // 4, index % 4])
        coordinates = coordinates_by_method[method]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=point_colors[method],
                   s=2.15, linewidths=0, alpha=1.0, rasterized=False)
        ax.set_title(method, fontsize=7.5, pad=1.5,
                     fontweight="bold" if method == "SpaCRAFT" else "normal",
                     color="#222222")
        ax.set_xticks([])
        ax.set_yticks([])
        if spatial:
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#777777")
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
            if index in (0, 4):
                add_umap_axes(ax)


def configure_figure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 8,
    })


def main():
    configure_figure_style()
    adata = ad.read_h5ad(DATA, backed="r")
    obs_names = adata.obs_names.astype(str)
    truth = adata.obs["Annotation_for_Combined"].astype(str).to_numpy()
    spatial = np.asarray(adata.obsm["spatial"])
    colors = spatial_domain_colors(truth, spatial)
    truth_colors = [colors[value] for value in truth]

    spatial_coordinates = {}
    spatial_colors = {}
    umap_coordinates = {}
    umap_colors = {}
    for method in METHODS:
        cluster_path = CLUSTER_ROOT / method / "E13_5-S1_atac.csv"
        predicted = aligned_column(cluster_path, obs_names, "cluster_label").astype(str)
        matched = match_clusters(predicted, truth)
        spatial_coordinates[method] = spatial
        spatial_colors[method] = [colors[value] for value in matched]
        umap_coordinates[method] = aligned_umap(
            UMAP_ROOT / method / "E13_5-S1_atac.csv", obs_names
        )
        umap_colors[method] = truth_colors

    fig = plt.figure(figsize=(12.0, 14.2), facecolor="white")
    outer = fig.add_gridspec(
        4, 1, height_ratios=[0.91, 1.32, 1.18, 0.86],
        left=0.032, right=0.99, bottom=0.028, top=0.988, hspace=0.115,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.34, 0.66], wspace=0.014)

    plot_panel_a(fig, top[0], spatial, truth, colors)
    draw_refined_panel_b(fig=fig, spec=top[1])
    plot_comparison_grid(fig, outer[1], spatial_coordinates, spatial_colors,
                         spatial=True)
    plot_comparison_grid(fig, outer[2], umap_coordinates, umap_colors,
                         spatial=False)
    # Reserve a small blank band above panel e so it sits clear of panel d.
    panel_e_area = outer[3].subgridspec(
        2, 1, height_ratios=[0.10, 0.90], hspace=0.0
    )
    draw_ablation_panel(fig, panel_e_area[1])

    add_panel_label(fig, top[0], "a")
    add_panel_label(fig, outer[1], "c")
    add_panel_label(fig, outer[2], "d")
    add_panel_label(fig, panel_e_area[1], "e")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "fig2_single_slice_benchmark.pdf"
    png = OUTPUT_DIR / "fig2_single_slice_benchmark.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=350, facecolor="white")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()

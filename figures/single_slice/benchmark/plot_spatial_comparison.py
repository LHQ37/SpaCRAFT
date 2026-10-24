#!/usr/bin/env python3
"""Rebuild panel c: E13.5 eight-method spatial comparison.

The original June Adobe component and its ``combine`` inputs were deleted.
This reproducible reconstruction uses the complete E13.5 cluster labels from
the 2026-07-27 unified-oracle run. It is not the original June raster/PDF.
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from domain_palette import spatial_domain_colors

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/E13_5-S1_atac.h5ad"
INPUT_ROOT = REPO / "results/single_slice/benchmark/oracle_cluster_labels"
OUT_DIR = Path(__file__).resolve().parent
METHODS = [
    ("SpaCRAFT", "SpaCRAFT"), ("INSTINCT", "INSTINCT"),
    ("Descart", "Descart"), ("SnapATAC2", "SnapATAC2"),
    ("epiScanpy", "epiScanpy"), ("PeakVI", "PeakVI"),
    ("SMOPCA", "SMOPCA"), ("SCALE", "SCALE"),
]


def aligned_column(path, obs_names):
    frame = pd.read_csv(path, dtype={"obs_name": str})
    if not {"obs_name", "cluster_label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain obs_name and cluster_label")
    if frame["obs_name"].duplicated().any():
        raise ValueError(f"duplicate obs_name values in {path}")
    values = frame.set_index("obs_name")["cluster_label"].reindex(obs_names)
    if values.isna().any():
        raise ValueError(f"{path} is missing {int(values.isna().sum())} spots")
    return values.astype(str).to_numpy()


def match_clusters(predicted, truth):
    pred_levels = pd.Categorical(predicted).categories.tolist()
    true_levels = pd.Categorical(truth).categories.tolist()
    table = np.zeros((len(pred_levels), len(true_levels)), dtype=int)
    pred_index = {value: i for i, value in enumerate(pred_levels)}
    true_index = {value: i for i, value in enumerate(true_levels)}
    for pred, true in zip(predicted, truth):
        table[pred_index[pred], true_index[true]] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {pred_levels[r]: true_levels[c] for r, c in zip(rows, cols)}
    for i, pred in enumerate(pred_levels):
        mapping.setdefault(pred, true_levels[int(np.argmax(table[i]))])
    return np.asarray([mapping[value] for value in predicted])


def main():
    adata = ad.read_h5ad(DATA, backed="r")
    obs_names = adata.obs_names.astype(str)
    truth = adata.obs["Annotation_for_Combined"].astype(str).to_numpy()
    coords = np.asarray(adata.obsm["spatial"])
    colors = spatial_domain_colors(truth, coords)

    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.45))
    for ax, (display_name, source_name) in zip(axes.flat, METHODS):
        source = INPUT_ROOT / source_name / "E13_5-S1_atac.csv"
        matched = match_clusters(aligned_column(source, obs_names), truth)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=[colors[value] for value in matched],
                   s=3.2, linewidths=0, alpha=1.0)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(display_name, fontsize=14, fontweight="normal", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#333333")

    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.96,
                        wspace=0.045, hspace=0.03)
    png = OUT_DIR / "spatial_comparison.png"
    pdf = OUT_DIR / "spatial_comparison.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

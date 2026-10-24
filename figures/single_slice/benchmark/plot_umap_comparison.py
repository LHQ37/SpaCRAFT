#!/usr/bin/env python3
"""Rebuild panel d: E13.5 eight-method UMAP comparison.

The original June Adobe component and its ``combine`` embeddings were deleted.
This reconstruction plots the complete 2026-07-27 unified-oracle UMAP
coordinates. UMAP1/UMAP2 arrows remain an Adobe assembly operation.
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from domain_palette import spatial_domain_colors

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/E13_5-S1_atac.h5ad"
INPUT_ROOT = REPO / "results/single_slice/benchmark/oracle_umap_coordinates"
OUT_DIR = Path(__file__).resolve().parent
METHODS = [
    ("SpaCRAFT", "SpaCRAFT"), ("INSTINCT", "INSTINCT"),
    ("Descart", "Descart"), ("SnapATAC2", "SnapATAC2"),
    ("epiScanpy", "epiScanpy"), ("PeakVI", "PeakVI"),
    ("SMOPCA", "SMOPCA"), ("SCALE", "SCALE"),
]


def aligned_coordinates(path, obs_names):
    frame = pd.read_csv(path, dtype={"obs_name": str})
    if not {"obs_name", "UMAP1", "UMAP2"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain obs_name, UMAP1 and UMAP2")
    if frame["obs_name"].duplicated().any():
        raise ValueError(f"duplicate obs_name values in {path}")
    coords = frame.set_index("obs_name")[["UMAP1", "UMAP2"]].reindex(obs_names)
    if coords.isna().any().any():
        raise ValueError(f"{path} has missing E13.5 coordinates")
    return coords.to_numpy()


def main():
    adata = ad.read_h5ad(DATA, backed="r")
    obs_names = adata.obs_names.astype(str)
    labels = adata.obs["Annotation_for_Combined"].astype(str)
    spatial = adata.obsm["spatial"]
    color_map = spatial_domain_colors(labels, spatial)
    point_colors = [color_map[value] for value in labels]

    fig, axes = plt.subplots(2, 4, figsize=(10.6, 4.85))
    for ax, (display_name, source_name) in zip(axes.flat, METHODS):
        source = INPUT_ROOT / source_name / "E13_5-S1_atac.csv"
        coords = aligned_coordinates(source, obs_names)
        ax.scatter(coords[:, 0], coords[:, 1], c=point_colors,
                   s=3.2, linewidths=0, alpha=1.0)
        ax.set_title(display_name, fontsize=14, fontweight="normal", pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.015, top=0.955,
                        wspace=0.06, hspace=0.04)
    png = OUT_DIR / "umap_comparison.png"
    pdf = OUT_DIR / "umap_comparison.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

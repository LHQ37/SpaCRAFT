#!/usr/bin/env python3
"""
Plot only the ground-truth spatial distribution for the benchmark dataset.

This script is split out from `plot_spatial_distribution_comparison.py` so that
true labels can be rendered independently from algorithm predictions.
"""

from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from domain_palette import spatial_domain_colors

DATA_PATH = REPO_ROOT / "data/E13_5-S1_atac.h5ad"
OUT_DIR = Path(__file__).resolve().parent
DATASET_NAME = 'E13_5-S1_atac'
LABEL_CANDIDATES = [
    'Annotation_for_Combined', 'cell_type', 'ATAC_clusters', 'ATAC_Clusters',
    'RNA_clusters', 'clusters', 'Annotation', 'annotation', 'label', 'cluster'
]


def find_label_column(adata):
    preferred = 'Annotation_for_Combined'
    if preferred in adata.obs.columns:
        return preferred
    for col in LABEL_CANDIDATES:
        if col in adata.obs.columns:
            return col
    categorical = adata.obs.select_dtypes(include=['object', 'category']).columns
    if len(categorical) > 0:
        return categorical[0]
    raise ValueError(f'未找到真实标签列，尝试过: {LABEL_CANDIDATES}')


def get_spatial_coords(adata):
    if 'spatial' in adata.obsm:
        return np.asarray(adata.obsm['spatial'])
    if {'array_col', 'array_row'}.issubset(adata.obs.columns):
        return adata.obs[['array_col', 'array_row']].to_numpy()
    if {'spatial_X', 'spatial_Y'}.issubset(adata.obs.columns):
        return adata.obs[['spatial_X', 'spatial_Y']].to_numpy()
    if {'X', 'Y'}.issubset(adata.obs.columns):
        return adata.obs[['X', 'Y']].to_numpy()
    raise ValueError("未找到空间坐标：需要 obsm['spatial'] 或 obs 中的 array_col/array_row 等列")


def try_plot_background(ax, adata):
    spatial_uns = adata.uns.get('spatial', None)
    if not isinstance(spatial_uns, dict) or len(spatial_uns) == 0:
        return False

    for library in spatial_uns.values():
        if not isinstance(library, dict):
            continue
        images = library.get('images', {})
        if 'hires' in images:
            ax.imshow(images['hires'], origin='upper')
            return True
        if 'lowres' in images:
            ax.imshow(images['lowres'], origin='upper')
            return True
    return False


def plot_panel(ax, coords, labels, color_map, title, adata):
    has_bg = try_plot_background(ax, adata)
    colors = [color_map.get(label, (0.7, 0.7, 0.7, 1.0)) for label in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, linewidths=0, alpha=1.0)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal', adjustable='box')
    if not has_bg:
        ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 80)
    print('Spatial distribution plot: ground truth only')
    print('=' * 80)
    print(f'Dataset: {DATA_PATH}')
    print(f'Output dir: {OUT_DIR}')

    adata = ad.read_h5ad(DATA_PATH)
    label_key = find_label_column(adata)
    true_labels = adata.obs[label_key].astype(str).to_numpy()
    coords = get_spatial_coords(adata)
    color_map = spatial_domain_colors(true_labels, coords)

    print(f'Label column: {label_key}')
    print(f'n_cells: {adata.n_obs}')
    print(f'n_labels: {len(np.unique(true_labels))}')
    print(f'spatial shape: {coords.shape}')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.4))
    plot_panel(ax, coords, true_labels, color_map, 'Ground Truth', adata)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=color, markersize=7)
        for cat, color in color_map.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.005, 0.5),
        frameon=False,
        title=label_key,
        fontsize=8,
        title_fontsize=9,
    )

    fig.suptitle(f'Ground truth spatial distribution: {DATASET_NAME}', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    png_path = OUT_DIR / 'ground_truth.png'
    pdf_path = OUT_DIR / 'ground_truth.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print('\n完成')
    print(f'PNG: {png_path}')
    print(f'PDF: {pdf_path}')


if __name__ == '__main__':
    main()

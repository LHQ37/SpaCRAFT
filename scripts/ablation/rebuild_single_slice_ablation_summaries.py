#!/usr/bin/env python3
"""Rebuild the formal loss-ablation summaries from per-dataset metrics."""

from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/ablation/single_slice_loss_ablation"
GRAPH_MEAN = REPO / "results/ablation/single_slice_graph_ablation_mean.csv"
PER_DATASET = ROOT / "single_slice_loss_ablation_per_dataset.csv"
MEAN = ROOT / "single_slice_loss_ablation_mean.csv"
MEAN_STD = ROOT / "single_slice_loss_ablation_mean_std.csv"

VARIANTS = [
    "Full", "w/o Reconstruction", "w/o InfoNCE", "w/o Adversarial",
]
METRICS = [
    "ARI", "NMI", "AMI", "FMI", "Homogeneity", "Completeness", "V-measure",
]


def main():
    graph = pd.read_csv(GRAPH_MEAN)
    if not np.allclose(graph["V-measure"], graph["NMI"], rtol=0.0, atol=1e-12):
        raise ValueError("graph mean V-measure must match arithmetic-normalized NMI")

    frame = pd.read_csv(PER_DATASET)
    required = {"dataset", "variant", *METRICS}
    if set(frame.columns) != required:
        raise ValueError(f"unexpected per-dataset columns: {list(frame.columns)}")
    if frame.groupby("variant")["dataset"].nunique().reindex(VARIANTS).tolist() != [6] * 4:
        raise ValueError("each formal loss-ablation variant must contain six datasets")
    if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
        raise ValueError("non-finite loss-ablation metric")
    h = frame["Homogeneity"].to_numpy(dtype=float)
    c = frame["Completeness"].to_numpy(dtype=float)
    expected_v = np.divide(2.0 * h * c, h + c, out=np.ones_like(h), where=(h + c) != 0)
    if not np.allclose(frame["V-measure"], expected_v, rtol=0.0, atol=1e-12):
        raise ValueError("V-measure is inconsistent with homogeneity/completeness")

    means = frame.groupby("variant", sort=False)[METRICS].mean().reindex(VARIANTS)
    means.index.name = "variant"
    means.to_csv(MEAN)

    rows = []
    for metric in METRICS:
        grouped = frame.groupby("variant", sort=False)[metric]
        for variant in VARIANTS:
            values = grouped.get_group(variant).to_numpy(dtype=float)
            rows.append({
                "variant": variant,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "n": int(values.size),
            })
    pd.DataFrame(rows).to_csv(MEAN_STD, index=False)
    print(MEAN)
    print(MEAN_STD)


if __name__ == "__main__":
    main()

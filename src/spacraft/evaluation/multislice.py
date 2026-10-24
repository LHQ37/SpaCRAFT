"""Shared data-loading and evaluation helpers for official multi-slice runs."""

from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chisquare
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


SOURCE_ROOT = Path("data/multi_slice/e11_e18")
SLICE_NAMES = ["E11_S1", "E13_S1", "E15_S1", "E18_S1"]
GAT_PARAMS = {
    "latent_dim": 10,
    "lambda_adv": 0.28,
    "lambda_rec": 10.0,
    "lambda_cluster_contrast": 1.1,
    "learn_rate": 0.00011929202947805012,
}


def required_inputs(source_root=SOURCE_ROOT):
    files = [
        source_root / "preprocessed_concat_atac.h5ad",
        source_root / "input_matrix_lsi_atac.npy",
        source_root / "feature_spatial_graph_list.npz",
    ]
    files.extend(source_root / f"prepared_{name}_atac.h5ad" for name in SLICE_NAMES)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    return files


def load_graph_list(path):
    archive = np.load(path)
    keys = sorted(archive.files, key=lambda key: int(key.rsplit("_", 1)[1]))
    return [np.asarray(archive[key]) for key in keys]


def load_inputs(smoke_spots_per_slice=0, source_root=SOURCE_ROOT):
    required_inputs(source_root)
    concat = ad.read_h5ad(source_root / "preprocessed_concat_atac.h5ad")
    slices = [
        ad.read_h5ad(source_root / f"prepared_{name}_atac.h5ad")
        for name in SLICE_NAMES
    ]
    raw = np.load(source_root / "input_matrix_lsi_atac.npy")
    if raw.shape != (concat.n_obs, 29):
        raw = TruncatedSVD(
            n_components=30, random_state=1234, n_iter=7
        ).fit_transform(concat.X)[:, 1:]
    raw = np.asarray(raw).copy()
    if not concat.obs_names.is_unique:
        raise ValueError("Concatenated obs_names must be unique")
    if "slice_name" not in concat.obs:
        raise ValueError("Concatenated input lacks obs['slice_name']")
    expected_names = np.concatenate([np.asarray(item.obs_names) for item in slices])
    if not np.array_equal(np.asarray(concat.obs_names), expected_names):
        raise ValueError("Slice and concatenated observation order differ")

    if smoke_spots_per_slice > 0:
        keep, smoke_slices, offset = [], [], 0
        for item in slices:
            n_spots = min(smoke_spots_per_slice, item.n_obs)
            keep.extend(range(offset, offset + n_spots))
            smoke_slices.append(item[:n_spots].copy())
            offset += item.n_obs
        keep = np.asarray(keep, dtype=int)
        concat = concat[keep].copy()
        slices = smoke_slices
        raw = raw[keep].copy()

    concat.obsm["X_lsi_raw"] = raw
    offset = 0
    for item in slices:
        item.obsm["X_lsi_raw"] = raw[offset:offset + item.n_obs].copy()
        offset += item.n_obs
    return slices, concat, raw


def encode_labels(values):
    return LabelEncoder().fit_transform(np.asarray(values).astype(str))


def label_column(adata):
    candidates = ["region", "annotation", "cell_type", "celltype",
                  "ground_truth", "label"]
    return next((key for key in candidates if key in adata.obs), None)


def batch_asw(embedding, batch):
    try:
        return 1.0 - abs(float(silhouette_score(embedding, batch)))
    except Exception:
        return None


def graph_connectivity(embedding, labels, n_neighbors=15):
    k = min(n_neighbors + 1, len(embedding))
    indices = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(
        embedding, return_distance=False
    )
    rows = np.repeat(np.arange(len(embedding)), k - 1)
    cols = indices[:, 1:].reshape(-1)
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(embedding), len(embedding)),
    ).maximum(
        csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(embedding), len(embedding)),
        ).T
    )
    scores = []
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        if len(idx) > 1:
            _, components = connected_components(
                graph[idx][:, idx], directed=False
            )
            scores.append(np.bincount(components).max() / len(idx))
    return float(np.mean(scores)) if scores else None


def batch_pcr(embedding, batch):
    n_pcs = min(20, embedding.shape[1], embedding.shape[0] - 1)
    if n_pcs < 1:
        return None
    pcs = PCA(n_components=n_pcs, random_state=0).fit_transform(embedding)
    encoded = encode_labels(batch)
    onehot = np.eye(len(np.unique(encoded)))[encoded]
    variances = np.var(pcs, axis=0)
    explained = [
        np.clip(
            LinearRegression().fit(onehot, pcs[:, i]).score(onehot, pcs[:, i]),
            0,
            1,
        )
        for i in range(n_pcs)
    ]
    return 1.0 - float(
        np.sum(np.asarray(explained) * variances) / np.sum(variances)
    )


def kbet(embedding, batch, k=50, alpha=0.05):
    encoded = encode_labels(batch)
    frequencies = np.bincount(encoded).astype(float)
    frequencies /= frequencies.sum()
    k = min(k, len(embedding) - 1)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(
        embedding, return_distance=False
    )[:, 1:]
    expected = frequencies * k
    rejected = tested = 0
    for row in indices:
        observed = np.bincount(
            encoded[row], minlength=len(frequencies)
        ).astype(float)
        valid = expected > 0
        if valid.sum() > 1:
            _, pvalue = chisquare(observed[valid], expected[valid])
            rejected += int(pvalue < alpha)
            tested += 1
    return 1.0 - rejected / tested if tested else None

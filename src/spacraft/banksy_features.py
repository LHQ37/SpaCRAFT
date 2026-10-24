"""BANKSY-inspired spatial features for multi-slice Raw spacraft.

This is deliberately a feature-only implementation: its within-slice KNNs are
never exposed as the graph consumed by the SpaCRAFT encoder.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class BanksyConfig:
    kgeom: int = 15
    lambda_banksy: float = 0.2
    mu: float = 1.5
    agf_multiplier: float = 2.0
    gaussian_scale_fraction: float = 0.5
    normalization_strategy: str = "per_slice_component_zscore"
    eps: float = 1e-8


def _zscore_columns(values, eps):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, eps)


def _knn(coordinates, n_neighbors):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("spatial coordinates must have shape (n_spots, >=2)")
    if coordinates.shape[0] <= n_neighbors:
        raise ValueError(
            f"BANKSY requires more than {n_neighbors} spots per slice; "
            f"received {coordinates.shape[0]}")
    # Ask for one extra result and explicitly discard self.
    distances, indices = NearestNeighbors(
        n_neighbors=n_neighbors + 1, metric="euclidean"
    ).fit(coordinates[:, :2]).kneighbors(coordinates[:, :2])
    keep_distances = np.empty((len(coordinates), n_neighbors), dtype=np.float64)
    keep_indices = np.empty((len(coordinates), n_neighbors), dtype=np.int64)
    for row in range(len(coordinates)):
        nonself = indices[row] != row
        keep_distances[row] = distances[row, nonself][:n_neighbors]
        keep_indices[row] = indices[row, nonself][:n_neighbors]
    return keep_distances, keep_indices


def _gaussian_weights(distances, scale_rank, eps):
    # scale_rank is one-based, matching "the floor(kgeom/2)-th neighbour".
    scale = distances[:, scale_rank - 1:scale_rank]
    scale = np.maximum(scale, eps)
    weights = np.exp(-0.5 * np.square(distances / scale))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)


def _compute_slice_components(features, coordinates, config):
    """Return unstandardized C, M, G and numerical audit data."""
    x = np.asarray(features, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)[:, :2]
    if x.ndim != 2 or len(x) != len(coords) or not np.isfinite(x).all():
        raise ValueError("features must be a finite (n_spots, n_features) matrix")

    kagf = max(config.kgeom, int(round(config.agf_multiplier * config.kgeom)))
    distances, neighbours = _knn(coords, kagf)
    scale_rank = min(
        kagf, max(1, int(config.gaussian_scale_fraction * config.kgeom)))
    weights = _gaussian_weights(distances, scale_rank, config.eps)

    mean_neighbours = neighbours[:, :config.kgeom]
    mean_weights = weights[:, :config.kgeom]
    # Renormalize after restricting the 2*kgeom query to the M neighbourhood.
    mean_weights /= np.maximum(mean_weights.sum(axis=1, keepdims=True), config.eps)
    m = np.einsum("nk,nkd->nd", mean_weights, x[mean_neighbours])

    offsets = coords[neighbours] - coords[:, None, :]
    phi = np.arctan2(offsets[:, :, 1], offsets[:, :, 0])
    real = np.einsum("nk,nkd->nd", weights * np.cos(phi), x[neighbours])
    imag = np.einsum("nk,nkd->nd", weights * np.sin(phi), x[neighbours])
    g = np.sqrt(np.square(real) + np.square(imag))

    theta = 0.731
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    rotated = coords @ rotation.T
    rotated_offsets = rotated[neighbours] - rotated[:, None, :]
    rotated_phi = np.arctan2(rotated_offsets[:, :, 1], rotated_offsets[:, :, 0])
    rotated_real = np.einsum(
        "nk,nkd->nd", weights * np.cos(rotated_phi), x[neighbours])
    rotated_imag = np.einsum(
        "nk,nkd->nd", weights * np.sin(rotated_phi), x[neighbours])
    rotated_g = np.sqrt(np.square(rotated_real) + np.square(rotated_imag))
    rotation_error = np.abs(rotated_g - g)

    def stats(values):
        return {
            "mean": float(values.mean()), "std": float(values.std()),
            "frobenius_norm": float(np.linalg.norm(values)),
        }

    audit = {
        "n_spots": int(len(x)),
        "kgeom_mean": config.kgeom,
        "kgeom_agf": kagf,
        "mean_neighbour_distance": float(distances[:, :config.kgeom].mean()),
        "min_neighbour_distance": float(distances[:, :config.kgeom].min()),
        "max_neighbour_distance": float(distances[:, :config.kgeom].max()),
        "local_scale_rank_one_based": scale_rank,
        "M": stats(m),
        "G": stats(g),
        "rotation_invariance": {
            "angle_radians": theta,
            "max_absolute_error": float(rotation_error.max()),
            "mean_absolute_error": float(rotation_error.mean()),
            "passed": bool(np.allclose(rotated_g, g, rtol=1e-6, atol=1e-8)),
        },
        "dimensions": {"C": x.shape[1], "M": m.shape[1], "G": g.shape[1],
                       "fused_before_projection": 3 * x.shape[1]},
    }
    return x, m, g, audit


def _fuse_components(c, m, g, config):
    return np.concatenate([
        np.sqrt(1.0 - config.lambda_banksy) * c,
        np.sqrt(config.lambda_banksy / config.mu) * m,
        np.sqrt(config.lambda_banksy / (2.0 * config.mu)) * g,
    ], axis=1)


def compute_slice_features(features, coordinates, config=BanksyConfig()):
    """Return fused, per-slice component-standardized BANKSY features."""
    x, m, g, audit = _compute_slice_components(features, coordinates, config)
    fused = _fuse_components(
        _zscore_columns(x, config.eps),
        _zscore_columns(m, config.eps),
        _zscore_columns(g, config.eps), config)
    return fused.astype(np.float32), audit


def build_banksy_features(adata_list, adata_concat, input_mat_key,
                          config=BanksyConfig()):
    """Build concatenated features using only neighbours from the same slice."""
    valid_strategies = {
        "per_slice_component_zscore", "global_component_zscore",
        "per_slice_component_zscore_plus_post_concat_layernorm",
        "per_slice_component_zscore_plus_post_projection_layernorm",
    }
    if config.normalization_strategy not in valid_strategies:
        raise ValueError(
            f"unknown BANKSY normalization strategy "
            f"{config.normalization_strategy!r}")
    source = np.asarray(adata_concat.obsm[input_mat_key])
    pieces, audits, components = [], [], []
    offset = 0
    for slice_id, adata in enumerate(adata_list):
        end = offset + adata.n_obs
        if "spatial" not in adata.obsm:
            raise KeyError(f"slice {slice_id} has no obsm['spatial']")
        if config.normalization_strategy == "global_component_zscore":
            c, m, g, audit = _compute_slice_components(
                source[offset:end], adata.obsm["spatial"], config)
            components.append((c, m, g))
            piece = None
        else:
            piece, audit = compute_slice_features(
                source[offset:end], adata.obsm["spatial"], config)
        audit["slice_id"] = slice_id
        audit["cross_slice_edges"] = 0
        if piece is not None:
            pieces.append(piece)
        audits.append(audit)
        offset = end
    if offset != len(source):
        raise ValueError("adata_list spot counts do not match adata_concat")
    if config.normalization_strategy == "global_component_zscore":
        sizes = [len(values[0]) for values in components]
        global_components = [
            _zscore_columns(np.vstack([values[i] for values in components]),
                            config.eps)
            for i in range(3)]
        fused = _fuse_components(*global_components, config)
        boundaries = np.cumsum([0] + sizes)
        pieces = [
            fused[boundaries[i]:boundaries[i + 1]].astype(np.float32)
            for i in range(len(sizes))]
    return np.vstack(pieces), {
        "method": "BANKSY-inspired spatial feature augmentation on Raw SpaCRAFT",
        "lambda_banksy": config.lambda_banksy,
        "mu": config.mu,
        "kgeom": config.kgeom,
        "agf_multiplier": config.agf_multiplier,
        "agf_neighbors": max(
            config.kgeom, int(round(config.agf_multiplier * config.kgeom))),
        "gaussian_scale_fraction": config.gaussian_scale_fraction,
        "normalization_strategy": config.normalization_strategy,
        "cross_slice_edges": 0,
        "raw_graph_replaced": False,
        "slices": audits,
    }

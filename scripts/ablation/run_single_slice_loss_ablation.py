#!/usr/bin/env python3
"""Formal leave-one-loss-out runner for the six single-slice benchmarks.

This adapts tests/run_spacraft_multi_datasets.py only for isolated output and
complete provenance. It does not alter the model, preprocessing, graph,
training parameters, or oracle Leiden search.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

import spacraft
from spacraft.model import SpaCRAFTModel

SEED = 1236
PARAMS = {
    "lsi_dim": 20,
    "latent_dim": 10,
    "hidden_dims_G": [50],
    "hidden_dims_D": [50],
    "lambda_cluster_contrast": 1.1,
    "lambda_adv": 0.28,
    "lambda_rec": 6.13,
    "contrastive_temperature": 0.5,
    "graph_k": 40,
    "spatial_weight": 0.82,
    "rad_coef": 1.5,
    "training_steps": 500,
    "learn_rate_G": 0.00011929202947805012,
    "learn_rate_D": 0.00005964601473902506,
    "similarity_metric": "cosine",
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "scheduler": "none",
    "feature_dropout": 0.0,
    "attention_dropout": 0.0,
    "gat_normalization": "none",
}
DATASETS = {
    "E11_0-S1_atac": ("E11_0-S1_atac.h5ad", "Annotation_for_Combined"),
    "E13_5-S1_atac": ("E13_5-S1_atac.h5ad", "Annotation_for_Combined"),
    "E15_5-S1_atac": ("E15_5-S1_atac.h5ad", "Annotation_for_Combined"),
    "E18_5-S1_atac": ("E18_5-S1_atac.h5ad", "Annotation_for_Combined"),
    "adata_peaks_normalized": ("adata_peaks_normalized.h5ad", "ATAC_clusters"),
    "s1_adata_atac_with_meta": ("s1_adata_atac_with_meta.h5ad", "ATAC_clusters"),
}
METRICS = [
    "ARI", "NMI", "AMI", "Homogeneity", "Completeness", "V-measure", "FMI"
]


def set_seed():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def metrics(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
        "FMI": fowlkes_mallows_score(y_true, y_pred),
    }


def normalize_peak_names(adata):
    names = []
    for peak in adata.var_names.astype(str):
        peak = peak.strip()
        if ":" in peak and "-" in peak:
            chrom, rest = peak.split(":", 1)
            start, end = rest.split("-", 1)
        elif peak.count("-") >= 2:
            chrom, start, end = peak.rsplit("-", 2)
        elif peak.count("_") >= 2:
            chrom, start, end = peak.rsplit("_", 2)
        else:
            raise ValueError(f"Unparseable peak name: {peak}")
        names.append(
            f"{chrom}:{int(float(start.replace(',', '')))}-"
            f"{int(float(end.replace(',', '')))}"
        )
    result = adata.copy()
    result.var_names = names
    result.var_names_make_unique()
    return result


def ensure_spatial(adata_concat, cas_list):
    def coordinates(value):
        if "spatial" in value.obsm:
            return np.asarray(value.obsm["spatial"])
        for cols in (
            ("array_row", "array_col"), ("spatial_X", "spatial_Y"), ("X", "Y")
        ):
            if set(cols).issubset(value.obs.columns):
                return value.obs[list(cols)].to_numpy()
        return None

    concat_coords = coordinates(adata_concat)
    if concat_coords is None:
        raise ValueError("No spatial coordinates")
    adata_concat.obsm["spatial"] = concat_coords
    for cas in cas_list:
        current = coordinates(cas)
        if current is None:
            positions = adata_concat.obs_names.get_indexer(cas.obs_names)
            if np.any(positions < 0):
                raise ValueError("Cannot align spatial coordinates")
            current = concat_coords[positions]
        cas.obsm["spatial"] = current


def oracle_leiden(embedding, adata_concat, truth, result_dir):
    work = adata_concat.copy()
    work.obsm["emb"] = embedding
    sc.pp.neighbors(
        work, n_neighbors=50, use_rep="emb", metric="euclidean",
        random_state=SEED,
    )
    tested = {}
    best_ari = -1.0
    best_resolution = None

    def evaluate(resolution):
        nonlocal best_ari, best_resolution
        resolution = round(float(resolution), 2)
        if resolution in tested:
            return tested[resolution]
        sc.tl.leiden(
            work, random_state=SEED, resolution=resolution,
            key_added="leiden_temp",
        )
        labels = work.obs["leiden_temp"].astype(str).to_numpy()
        values = metrics(truth, labels)
        tested[resolution] = {
            "resolution": resolution,
            "n_clusters": int(np.unique(labels).size),
            **values,
            "_labels": labels.copy(),
        }
        if (
            values["ARI"] > best_ari
            or (
                np.isclose(values["ARI"], best_ari, rtol=0.0, atol=0.0)
                and (best_resolution is None or resolution < best_resolution)
            )
        ):
            best_ari = values["ARI"]
            best_resolution = resolution
        return tested[resolution]

    for resolution in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        evaluate(resolution)
    gaps = [
        (0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.0),
        (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0),
    ]
    for start, end in gaps:
        start_result = tested.get(start)
        end_result = tested.get(end)
        if (
            start_result and end_result
            and max(start_result["ARI"], end_result["ARI"]) < best_ari - 0.05
        ):
            continue
        left, right = start, end
        for _ in range(4):
            midpoint = round((left + right) / 2, 2)
            if any(abs(midpoint - old) < 0.02 for old in tested):
                break
            result = evaluate(midpoint)
            if result["ARI"] > (start_result["ARI"] if start_result else 0):
                left = midpoint
            else:
                right = midpoint
    for offset in [-0.15, -0.1, -0.05, 0.05, 0.1, 0.15]:
        resolution = round(best_resolution + offset, 2)
        if resolution > 0:
            evaluate(resolution)

    selected = tested[best_resolution]
    labels = selected["_labels"]
    candidate_rows = [
        {key: value for key, value in tested[resolution].items()
         if key != "_labels"}
        for resolution in sorted(tested)
    ]
    pd.DataFrame(candidate_rows).to_csv(
        result_dir / "all_resolution_candidates.csv", index=False
    )
    pd.DataFrame({
        "obs_name": adata_concat.obs_names.astype(str),
        "cluster_label": labels.astype(str),
    }).to_csv(result_dir / "cluster_labels.csv", index=False)
    selected_metrics = {key: selected[key] for key in METRICS}
    selected_metrics["Overall"] = float(
        np.mean([selected_metrics[key] for key in METRICS])
    )
    pd.DataFrame([selected_metrics]).to_csv(result_dir / "metrics.csv", index=False)
    with open(result_dir / "selected_resolution.json", "w") as handle:
        json.dump({
            "selected_resolution": best_resolution,
            "selected_ARI": best_ari,
            "n_clusters": selected["n_clusters"],
            "tie_rule": "smaller resolution wins exact ARI ties",
        }, handle, indent=2)

    work.obs["selected_clusters"] = labels
    sc.tl.umap(work, random_state=SEED)
    umap = np.asarray(work.obsm["X_umap"])
    pd.DataFrame({
        "obs_name": adata_concat.obs_names.astype(str),
        "UMAP1": umap[:, 0],
        "UMAP2": umap[:, 1],
    }).to_csv(result_dir / "umap_coordinates.csv", index=False)
    with open(result_dir / "leiden_metadata.json", "w") as handle:
        json.dump({
            "scanpy_version": sc.__version__,
            "n_neighbors": 50,
            "metric": "euclidean",
            "random_state": SEED,
            "umap_random_state": SEED,
            "search_mode": "thorough",
            "candidate_count": len(tested),
            "selected_resolution": best_resolution,
            "selection_metric": "ARI",
        }, handle, indent=2)
    return selected_metrics, best_resolution


def sha256_outputs(result_dir):
    rows = []
    for path in sorted(result_dir.iterdir()):
        if path.is_file() and path.name != "sha256.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.name}")
    (result_dir / "sha256.txt").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--input-root", type=Path, default=REPO / "data")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "results/ablation/single_slice_loss_ablation")
    parser.add_argument("--variant", choices=["full", "wo_reconstruction", "wo_infonce", "wo_adversarial"], required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--validation-run", action="store_true")
    args = parser.parse_args()
    run_root = (args.output_root / "validation" / args.variant
                if args.validation_run else args.output_root / args.variant)
    result_dir = run_root / args.dataset
    result_dir.mkdir(parents=True, exist_ok=False)
    command = " ".join(sys.argv)
    (result_dir / "actual_command.txt").write_text(command + "\n")

    start = time.time()
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("Formal run requires CUDA visibility")
    device = torch.device("cuda:0")
    filename, label_key = DATASETS[args.dataset]
    data_path = args.input_root / filename
    adata = ad.read_h5ad(data_path)
    adata.var_names_make_unique()
    if label_key not in adata.obs:
        raise ValueError(f"Missing locked label column {label_key}")
    truth = adata.obs[label_key].astype(str).to_numpy()
    original_shape = list(adata.shape)
    adata = normalize_peak_names(adata)
    cas_list = spacraft.peak_sets_alignment([adata.copy()])
    adata_concat = ad.concat(cas_list, label="slice_name", keys=["sample"])
    spacraft.preprocess_CAS(
        cas_list, adata_concat, use_fragment_count=True, min_cells_rate=0.03
    )
    lsi = TruncatedSVD(
        n_components=PARAMS["lsi_dim"], random_state=SEED, n_iter=7)
    embedding_lsi = lsi.fit_transform(adata_concat.X)[:, 1:]
    adata_concat.obsm["X_lsi"] = embedding_lsi
    ensure_spatial(adata_concat, cas_list)
    spacraft.create_neighbor_graph(
        cas_list, adata_concat, use_feature=True, feature_key="X_lsi",
        k=PARAMS["graph_k"],
        similarity_metric=PARAMS["similarity_metric"],
        spatial_weight=PARAMS["spatial_weight"],
        rad_coef=PARAMS["rad_coef"],
    )
    loss_weights = {
        "lambda_adv": 0.0 if args.variant == "wo_adversarial" else PARAMS["lambda_adv"],
        "lambda_rec": 0.0 if args.variant == "wo_reconstruction" else PARAMS["lambda_rec"],
        "lambda_cluster_contrast": 0.0 if args.variant == "wo_infonce" else PARAMS["lambda_cluster_contrast"],
    }
    model = SpaCRAFTModel(
        adata_list=cas_list,
        adata_concat=adata_concat,
        input_mat_key="X_lsi",
        input_dim=PARAMS["lsi_dim"] - 1,
        hidden_dims_G=PARAMS["hidden_dims_G"],
        latent_dim=PARAMS["latent_dim"],
        hidden_dims_D=PARAMS["hidden_dims_D"],
        lambda_adv=loss_weights["lambda_adv"],
        lambda_rec=loss_weights["lambda_rec"],
        lambda_cluster_contrast=loss_weights["lambda_cluster_contrast"],
        contrastive_temperature=PARAMS["contrastive_temperature"],
        learn_rates=[PARAMS["learn_rate_G"], PARAMS["learn_rate_D"]],
        steps=args.steps,
        enable_contrastive=True,
        seed=SEED,
        multi_slice_method="raw",
        optimizer_type=PARAMS["optimizer"].lower(),
        weight_decay=PARAMS["weight_decay"],
        scheduler_type=PARAMS["scheduler"],
        encoder_feature_dropout=PARAMS["feature_dropout"],
        attention_dropout=PARAMS["attention_dropout"],
        gat_normalization=PARAMS["gat_normalization"],
        device=device,
    )
    if model.multi_slice_method != "raw":
        raise RuntimeError("Single-slice effective mode is not Raw")
    if hasattr(model, "banksy_projection") or hasattr(model, "banksy_features"):
        raise RuntimeError("BANKSY object unexpectedly exists")
    model.train(report_loss=False, record_loss_history=True)
    model.encoder.eval()
    with torch.no_grad():
        latent = model.encoder(model.X, model.G).detach().cpu().numpy()
    if latent.shape != (adata_concat.n_obs, PARAMS["latent_dim"]):
        raise RuntimeError(f"Unexpected embedding shape {latent.shape}")
    if not np.isfinite(latent).all():
        raise RuntimeError("Embedding contains NaN/Inf")

    np.save(result_dir / "embedding.npy", latent)
    pd.DataFrame({"obs_name": adata_concat.obs_names.astype(str)}).to_csv(
        result_dir / "obs_names.csv", index=False
    )
    pd.DataFrame({
        "obs_name": adata_concat.obs_names.astype(str),
        "ground_truth": truth,
    }).to_csv(result_dir / "ground_truth_labels.csv", index=False)
    spatial = np.asarray(adata_concat.obsm["spatial"])
    pd.DataFrame({
        "obs_name": adata_concat.obs_names.astype(str),
        "spatial_1": spatial[:, 0],
        "spatial_2": spatial[:, 1],
    }).to_csv(result_dir / "spatial_coordinates.csv", index=False)
    pd.DataFrame(model.loss_history).to_csv(
        result_dir / "loss_history.csv", index=False
    )
    history = pd.DataFrame(model.loss_history)
    if not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Loss history contains NaN/Inf")
    zero_column = {
        "wo_reconstruction": "weighted_loss_rec",
        "wo_infonce": "weighted_loss_cluster_contrast",
        "wo_adversarial": "weighted_loss_adv_g",
    }.get(args.variant)
    if zero_column and not np.allclose(history[zero_column].to_numpy(), 0.0):
        raise RuntimeError(f"Disabled loss is not zero: {zero_column}")
    torch.save({
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "optimizer_G": model.optimizer_G.state_dict(),
        "optimizer_D": model.optimizer_D.state_dict(),
        "parameters": PARAMS,
        "seed": SEED,
    }, result_dir / "model_state.pt")

    selected_metrics, selected_resolution = oracle_leiden(
        latent, adata_concat, truth, result_dir
    )
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        source_commit = None
    metadata = {
        "status": "complete",
        "artifact_role": "formal SpaCRAFT single-slice loss ablation",
        "dataset": args.dataset,
        "variant": args.variant,
        "disabled_loss": {"full": None, "wo_reconstruction": "reconstruction",
                          "wo_infonce": "InfoNCE", "wo_adversarial": "adversarial"}[args.variant],
        "data_path": str(data_path),
        "original_shape": original_shape,
        "post_filter_shape": list(adata_concat.shape),
        "label_column": label_key,
        "true_cluster_count": int(pd.Series(truth).nunique()),
        "seed": SEED,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameters": PARAMS,
        "loss_weights": loss_weights,
        "training_steps": args.steps,
        "learning_rate": [PARAMS["learn_rate_G"], PARAMS["learn_rate_D"]],
        "graph_config": {"k": PARAMS["graph_k"], "spatial_weight": PARAMS["spatial_weight"],
                         "rad_coef": PARAMS["rad_coef"], "similarity_metric": PARAMS["similarity_metric"]},
        "validation_run": bool(args.validation_run),
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_commit_if_available": source_commit,
        "lsi_components_before_drop": PARAMS["lsi_dim"],
        "first_lsi_component_dropped": True,
        "model_input_dimension": PARAMS["lsi_dim"] - 1,
        "latent_dimension": PARAMS["latent_dim"],
        "effective_input_mode": model.multi_slice_method,
        "banksy_created": False,
        "projection_created": False,
        "n_cls": model.n_cls,
        "selected_resolution": selected_resolution,
        "metrics": selected_metrics,
        "embedding_shape": list(latent.shape),
        "embedding_dtype": str(latent.dtype),
        "embedding_finite": True,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - start,
    }
    with open(result_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    sha256_outputs(result_dir)
    print(json.dumps({
        "dataset": args.dataset,
        "status": "complete",
        "selected_resolution": selected_resolution,
        "metrics": selected_metrics,
        "elapsed_seconds": metadata["elapsed_seconds"],
    }))


if __name__ == "__main__":
    main()

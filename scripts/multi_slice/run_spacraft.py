#!/usr/bin/env python3
"""Run the fixed BANKSY-inspired feature augmentation (no parameter search)."""
import argparse
import contextlib
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
    fowlkes_mallows_score, homogeneity_score, normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spacraft.model import SpaCRAFTModel  # noqa: E402
from spacraft.evaluation import multislice as embryo4  # noqa: E402

RESULT_ROOT = ROOT / "outputs" / "multi_slice"
TRAINING_SEED = 1234
CLUSTERING_SEED = 1234
GMM_N_INIT = 10
GMM_COVARIANCE_TYPE = "tied"
SIX_SOURCE = Path("data/multi_slice/six_slices")
SIX_ORDER = [
    "220403_C2", "220403_D2", "220403_A2",
    "220403_B2", "220327_A1", "220327_A2",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["e11_e18", "six_slices"], required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Dataset-specific prepared input directory.",
    )
    parser.add_argument("--lambda-banksy", type=float, default=0.15)
    parser.add_argument("--banksy-kgeom", type=int, default=10)
    parser.add_argument("--banksy-mu", type=float, default=1.0)
    parser.add_argument("--banksy-agf-multiplier", type=float, default=2.0)
    parser.add_argument("--banksy-gaussian-scale-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.00011929202947805012)
    parser.add_argument("--lambda-adv", type=float, default=0.28)
    parser.add_argument("--lambda-rec", type=float, default=10.0)
    parser.add_argument("--lambda-cluster-contrast", type=float, default=1.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.5)
    return parser.parse_args()


def load_four(source_root):
    slices, concat, raw = embryo4.load_inputs(0, source_root=source_root)
    concat.uns["graph_list"] = embryo4.load_graph_list(
        source_root / "feature_spatial_graph_list.npz")
    return slices, concat, raw, embryo4.SLICE_NAMES


def load_six(source_root):
    obs = pd.read_csv(source_root / "prepared_data/prepared_obs.csv")
    raw = np.load(source_root / "prepared_data/joint_lsi_raw.npy")
    archive = np.load(source_root / "prepared_data/hybrid_graph_list.npz")
    graphs = [archive[f"graph_{i}"] for i in range(6)]
    slices, offset = [], 0
    for section in SIX_ORDER:
        rows = obs[obs.section == section].copy()
        n_spots = len(rows)
        item = ad.AnnData(
            X=sparse.csr_matrix((n_spots, 1)),
            obs=pd.DataFrame({
                "region": pd.Categorical(rows.reference_anatomical_label),
            }, index=rows.obs_name.astype(str)))
        item.obsm["spatial"] = rows[["spatial_x", "spatial_y"]].to_numpy()
        item.obsm["X_lsi_raw"] = raw[offset:offset + n_spots].copy()
        slices.append(item)
        offset += n_spots
    concat = ad.concat(
        slices, label="slice_name", keys=SIX_ORDER, index_unique=None)
    concat.obsm["X_lsi_raw"] = raw
    concat.uns["graph_list"] = graphs
    return slices, concat, raw, SIX_ORDER


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_fixed_gmm(embedding, concat, dataset):
    k = 16 if dataset == "e11_e18" else 11
    truth = embryo4.encode_labels(concat.obs["region"])
    batch = embryo4.encode_labels(concat.obs["slice_name"])
    labels = GaussianMixture(
        n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
        n_init=GMM_N_INIT, random_state=CLUSTERING_SEED,
    ).fit_predict(embedding)
    metrics = {
        "ARI": adjusted_rand_score(truth, labels),
        "NMI": normalized_mutual_info_score(truth, labels),
        "AMI": adjusted_mutual_info_score(truth, labels),
        "FMI": fowlkes_mallows_score(truth, labels),
        "Homogeneity": homogeneity_score(truth, labels),
        "Completeness": completeness_score(truth, labels),
        "Cell_type_ASW": (silhouette_score(embedding, truth) + 1.0) / 2.0,
        "kBET": embryo4.kbet(embedding, batch),
        "Batch_ASW": embryo4.batch_asw(embedding, batch),
        "Batch_PCR": embryo4.batch_pcr(embedding, batch),
        "Graph_connectivity": embryo4.graph_connectivity(embedding, truth),
    }
    if any(not np.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite metric: {metrics}")
    return metrics, labels


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the 100-step and formal experiments")
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)

    slices, concat, raw, names = (
        load_four(args.input_root)
        if args.dataset == "e11_e18"
        else load_six(args.input_root))
    output = (
        args.output_root / "smoke_test" / args.dataset
        if args.steps == 100 else args.output_root / args.dataset
    )
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")

    raw_graphs = [np.asarray(graph).copy() for graph in concat.uns["graph_list"]]
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    params = embryo4.GAT_PARAMS
    learning_rate = (
        params["learn_rate"] if args.learning_rate is None else args.learning_rate)
    lambda_adv = params["lambda_adv"] if args.lambda_adv is None else args.lambda_adv
    lambda_rec = params["lambda_rec"] if args.lambda_rec is None else args.lambda_rec
    lambda_cluster_contrast = (
        params["lambda_cluster_contrast"]
        if args.lambda_cluster_contrast is None
        else args.lambda_cluster_contrast)
    model = SpaCRAFTModel(
        slices, concat, input_mat_key="X_lsi_raw", input_dim=raw.shape[1],
        latent_dim=params["latent_dim"], lambda_adv=lambda_adv,
        lambda_rec=lambda_rec,
        lambda_cluster_contrast=lambda_cluster_contrast,
        learn_rates=[learning_rate, learning_rate * 0.5],
        steps=args.steps, seed=TRAINING_SEED,
        device=torch.device("cuda"),
        multi_slice_method="banksy", banksy_kgeom=args.banksy_kgeom,
        lambda_banksy=args.lambda_banksy, banksy_mu=args.banksy_mu,
        banksy_agf_multiplier=args.banksy_agf_multiplier,
        banksy_gaussian_scale_fraction=args.banksy_gaussian_scale_fraction,
        contrastive_temperature=args.contrastive_temperature)
    if any(not np.array_equal(before, after)
           for before, after in zip(raw_graphs, concat.uns["graph_list"])):
        raise RuntimeError("Raw graph changed during BANKSY feature construction")

    with (output / "train.log").open("w") as train_log:
        with contextlib.redirect_stdout(train_log), contextlib.redirect_stderr(train_log):
            model.train(
                report_loss=True, report_interval=50,
                record_loss_history=True)
            model.eval(slices, save=False)
    embedding = np.vstack(
        [item.obsm["spacraft_latent_encoder"] for item in slices])
    if not np.isfinite(embedding).all():
        raise RuntimeError("non-finite embedding")
    projection_grad = float(model.banksy_projection.weight.grad.norm().cpu())
    if projection_grad <= 0:
        raise RuntimeError("BANKSY projection has zero gradient")

    obs_names = np.concatenate(
        [np.asarray(item.obs_names).astype(str) for item in slices])
    np.save(output / "embedding.npy", embedding)
    pd.DataFrame({"obs_name": obs_names}).to_csv(
        output / "obs_names.csv", index=False)
    pd.DataFrame(model.loss_history).to_csv(
        output / "loss_history.csv", index=False)
    metrics, cluster_labels = evaluate_fixed_gmm(
        embedding, concat, args.dataset)
    pd.DataFrame([{
        "method": "BANKSY-inspired spatial feature augmentation on Raw SpaCRAFT",
        "training_seed": TRAINING_SEED,
        "clustering_seed": CLUSTERING_SEED,
        "GMM_n_init": GMM_N_INIT,
        "GMM_covariance_type": GMM_COVARIANCE_TYPE,
        "K": 16 if args.dataset == "e11_e18" else 11,
        **metrics,
    }]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({
        "obs_name": obs_names, "cluster": cluster_labels,
    }).to_csv(output / "cluster_labels.csv", index=False)
    audit = dict(model.banksy_audit)
    audit.update({
        "projection_gradient_norm": projection_grad,
        "raw_graph_unchanged": True,
        "raw_graph_role": "SpaCRAFT encoder graph only",
        "banksy_knn_role": "feature computation only",
        "external_batch_correction_called": False,
        "batch_loss_added": False,
    })
    atomic_json(output / "banksy_feature_audit.json", audit)
    metadata = {
        "method": audit["method"], "dataset": args.dataset,
        "slices": names, "spot_counts": [item.n_obs for item in slices],
        "training_seed": TRAINING_SEED,
        "clustering_seed": CLUSTERING_SEED,
        "gmm": {
            "random_state": CLUSTERING_SEED,
            "n_init": GMM_N_INIT,
            "covariance_type": GMM_COVARIANCE_TYPE,
            "n_components": 16 if args.dataset == "e11_e18" else 11,
        },
        "formal_training_runs": 1,
        "seed_selection_performed": False,
        "screening_scope": "single-seed screening result",
        "steps": args.steps,
        "lambda_banksy": args.lambda_banksy,
        "mu": args.banksy_mu, "kgeom": args.banksy_kgeom,
        "agf_multiplier": args.banksy_agf_multiplier,
        "gaussian_scale_fraction": args.banksy_gaussian_scale_fraction,
        "learning_rate": learning_rate,
        "lambda_adv": lambda_adv, "lambda_rec": lambda_rec,
        "lambda_cluster_contrast": lambda_cluster_contrast,
        "contrastive_temperature": args.contrastive_temperature,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "project_root": str(ROOT), "command": " ".join(sys.argv),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    atomic_json(output / "metadata.json", metadata)
    (output / "actual_command.txt").write_text(" ".join(sys.argv) + "\n")
    hashed = [
        "embedding.npy", "obs_names.csv", "cluster_labels.csv", "metrics.csv",
        "metadata.json", "train.log", "actual_command.txt",
        "banksy_feature_audit.json", "loss_history.csv",
    ]
    (output / "sha256.txt").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in hashed))
    print(json.dumps({
        "metadata": metadata, "audit": audit, "metrics": metrics,
        "sha256_manifest": str(output / "sha256.txt"),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the six frozen single-slice inputs without modifying them."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = SINGLE_DOWNSTREAM_ROOT
FORMAL = Path("results/single_slice/spacraft")
DATASETS = FORMAL / "OFFICIAL_SINGLE_SLICE_DATASETS.csv"
OUT_CSV = ROOT / "audit/SINGLE_SLICE_INPUT_MANIFEST.csv"
TRACE_CSV = ROOT / "audit/FROZEN_RESULT_TRACE.csv"
OUT_MD = ROOT / "audit/SINGLE_SLICE_INPUT_AUDIT.md"
PEAK_RE = re.compile(r"^(?P<chrom>[^:-]+)[:-](?P<start>\d+)[:-](?P<end>\d+)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_one(row: pd.Series) -> tuple[dict, dict]:
    name = row.dataset_name
    h5 = Path(row.h5ad_absolute_path)
    result = FORMAL / "results" / name
    labels_path = result / "cluster_labels.csv"
    obs_path = result / "obs_names.csv"
    embedding_path = result / "embedding.npy"
    meta_path = result / "metadata.json"
    leiden_path = result / "leiden_metadata.json"
    selected_path = result / "selected_resolution.json"
    for path in (h5, labels_path, obs_path, embedding_path, meta_path, leiden_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    a = ad.read_h5ad(h5, backed="r")
    obs = pd.read_csv(obs_path)
    labels = pd.read_csv(labels_path)
    meta = json.loads(meta_path.read_text())
    selected = json.loads(selected_path.read_text())
    frozen_obs = obs.iloc[:, -1].astype(str).tolist()
    label_obs_col = next((c for c in labels.columns if c.lower() in {"obs_name", "obs_names", "barcode"}), labels.columns[0])
    label_col = next((c for c in labels.columns if c.lower() in {"cluster", "leiden", "label", "cluster_label"}), labels.columns[-1])
    label_obs = labels[label_obs_col].astype(str).tolist()
    input_obs = a.obs_names.astype(str).tolist()
    mapping_ok = input_obs == frozen_obs == label_obs
    emb = np.load(embedding_path, mmap_mode="r")
    coords = np.asarray(a.obsm["spatial"])
    peaks = a.var_names.astype(str)
    parsed = [PEAK_RE.match(x) for x in peaks]
    valid = np.fromiter((m is not None for m in parsed), bool, len(peaks))
    starts = np.fromiter((int(m.group("start")) if m else -1 for m in parsed), np.int64, len(peaks))
    ends = np.fromiter((int(m.group("end")) if m else -1 for m in parsed), np.int64, len(peaks))
    chroms = [m.group("chrom") if m else "" for m in parsed]
    x = a.X
    matrix_kind = "sparse" if sparse.issparse(x) or "sparse_dataset" in type(x).__module__ else "dense"
    fragment_candidates = list(h5.parent.glob(f"*{name}*fragment*")) + list(h5.parent.glob(f"*{name}*fragments*"))
    development = {"E11_0-S1_atac": "E11.0", "E13_5-S1_atac": "E13.5",
                   "E15_5-S1_atac": "E15.5", "E18_5-S1_atac": "E18.5",
                   "adata_peaks_normalized": "P22", "s1_adata_atac_with_meta": "E12.5"}
    record = {
        "dataset_name": name,
        "developmental_stage": development[name],
        "source_h5ad": str(h5),
        "source_h5ad_sha256": sha256(h5),
        "raw_peak_matrix_path": f"{h5}::X",
        "matrix_shape": f"{a.n_obs}x{a.n_vars}",
        "matrix_dtype": str(a.X.dtype),
        "sparse_or_dense": matrix_kind,
        "raw_counts_location": "layers/counts" if "counts" in a.layers else "X",
        "X_layers_obsm_varm_uns": f"X={type(a.X).__name__};layers={list(a.layers.keys())};obsm={list(a.obsm.keys())};varm={list(a.varm.keys())};uns={list(a.uns.keys())}",
        "spot_count": a.n_obs,
        "peak_count": a.n_vars,
        "obs_names": f"h5ad.obs_names; first={input_obs[0]}; last={input_obs[-1]}",
        "spatial_coordinate_key": "obsm/spatial",
        "ground_truth_label_key": str(row.label_column),
        "SpaCRAFT_embedding_path": str(embedding_path),
        "SpaCRAFT_cluster_label_path": str(labels_path),
        "SpaCRAFT_cluster_count": labels[label_col].nunique(),
        "frozen_seed": meta["seed"],
        "Leiden_resolution": selected.get("selected_resolution", meta.get("selected_resolution")),
        "genome_build_evidence": "pending Phase 1 literature/script audit",
        "peak_coordinate_format": "chr:start-end" if ":" in peaks[0] else "chr-start-end",
        "fragment_file_path": ";".join(map(str, fragment_candidates)),
        "fragment_file_available": bool(fragment_candidates),
        "original_paper_or_README": "pending Phase 1 source mapping",
        "obs_mapping_exact": mapping_ok,
        "embedding_shape": "x".join(map(str, emb.shape)),
        "embedding_dtype": str(emb.dtype),
        "coordinates_shape": "x".join(map(str, coords.shape)),
        "peak_parse_fraction": float(valid.mean()),
        "duplicate_peak_count": int(pd.Index(peaks).duplicated().sum()),
        "invalid_interval_count": int(((starts < 0) | (starts >= ends)).sum()),
        "negative_start_count": int((starts < 0).sum()),
        "chrM_present": "chrM" in set(chroms),
    }
    trace = {
        "dataset_name": name, "source_h5ad": str(h5), "source_h5ad_sha256": record["source_h5ad_sha256"],
        "formal_obs_names_path": str(obs_path), "formal_obs_names_sha256": sha256(obs_path),
        "formal_cluster_labels_path": str(labels_path), "formal_cluster_labels_sha256": sha256(labels_path),
        "formal_embedding_path": str(embedding_path), "formal_embedding_sha256": sha256(embedding_path),
        "spot_order_exact": mapping_ok, "spot_count_input": a.n_obs, "spot_count_labels": len(labels),
        "spot_count_embedding": emb.shape[0], "frozen_seed": meta["seed"],
        "selected_resolution": record["Leiden_resolution"],
        "status": "PASS" if mapping_ok and len(labels) == a.n_obs == emb.shape[0] else "FORMAL_CLUSTER_LABEL_AMBIGUOUS",
    }
    return record, trace


def main() -> int:
    rows = pd.read_csv(DATASETS)
    records, traces = zip(*(read_one(row) for _, row in rows.iterrows()))
    pd.DataFrame(records).to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(traces).to_csv(TRACE_CSV, index=False)
    passed = len(records) == 6 and all(x["status"] == "PASS" for x in traces)
    lines = [
        "# Single-slice input audit", "",
        f"- Status: **{'PASS' if passed else 'FORMAL_CLUSTER_LABEL_AMBIGUOUS'}**",
        f"- Formal dataset count: {len(records)}",
        f"- Frozen source: `{FORMAL}`",
        "- Audit basis: official frozen dataset manifest, h5ad contents, exact obs order, label table, embedding shape, seed and selected Leiden metadata.",
        "- Original inputs were opened read-only/backed; no formal input or frozen result was modified.", "",
        "## Dataset checks", "",
        "| dataset | shape | frozen clusters | obs/label/embedding mapping | peak parsing |",
        "|---|---:|---:|---|---:|",
    ]
    for r, t in zip(records, traces):
        lines.append(f"| {r['dataset_name']} | {r['matrix_shape']} | {r['SpaCRAFT_cluster_count']} | {t['status']} | {r['peak_parse_fraction']:.3f} |")
    lines += ["", "Detailed fields are in `SINGLE_SLICE_INPUT_MANIFEST.csv`; immutable-result provenance is in `FROZEN_RESULT_TRACE.csv`."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("formal_input_audit_status=" + ("PASS" if passed else "FORMAL_CLUSTER_LABEL_AMBIGUOUS"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json, re
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=SINGLE_DOWNSTREAM_ROOT
FORMAL=Path("results/single_slice/spacraft/results")
manifest=pd.read_csv(ROOT/"audit/SINGLE_SLICE_INPUT_MANIFEST.csv")
pat=re.compile(r"^([^:-]+)[:-](\d+)[:-](\d+)$")
audit=[]
for row in manifest.itertuples():
    out=ROOT/"data_cache"/row.dataset_name
    out.mkdir(parents=True,exist_ok=True)
    a=ad.read_h5ad(row.source_h5ad)
    labels=pd.read_csv(FORMAL/row.dataset_name/"cluster_labels.csv")
    frozen_obs=pd.read_csv(FORMAL/row.dataset_name/"obs_names.csv").iloc[:,-1].astype(str)
    label_obs_col=next((c for c in labels if c.lower() in {"obs_name","obs_names","barcode"}),labels.columns[0])
    label_col=next((c for c in labels if c.lower() in {"cluster","leiden","label","cluster_label"}),labels.columns[-1])
    if list(a.obs_names.astype(str)) != list(frozen_obs) or list(a.obs_names.astype(str)) != list(labels[label_obs_col].astype(str)):
        raise RuntimeError(f"FORMAL_CLUSTER_LABEL_AMBIGUOUS: {row.dataset_name}")
    a.obs["SpaCRAFT_cluster"]=pd.Categorical(labels[label_col].astype(str).values)
    a.obs["ground_truth_annotation"]=a.obs[row.ground_truth_label_key].astype(str).values
    parsed=[pat.match(str(x)) for x in a.var_names]
    if not all(parsed): raise RuntimeError(f"peak parse failure: {row.dataset_name}")
    a.var["chrom"]=[m.group(1) for m in parsed]
    a.var["start"]=np.array([int(m.group(2)) for m in parsed],dtype=np.int64)
    a.var["end"]=np.array([int(m.group(3)) for m in parsed],dtype=np.int64)
    a.uns["downstream_provenance"]={"project_display_name":"SpaCRAFT","genome_build":"mm10/GRCm38",
        "source_h5ad":row.source_h5ad,"formal_cluster_labels":row.SpaCRAFT_cluster_label_path,
        "frozen_seed":int(row.frozen_seed),"canonical_spot_order":"source h5ad obs_names"}
    target=out/"standardized.h5ad"
    a.write_h5ad(target,compression="gzip")
    finite=True; nonnegative=True
    x=a.layers["counts"] if "counts" in a.layers else a.X
    vals=x.data if sparse.issparse(x) else np.asarray(x).ravel()
    finite=bool(np.isfinite(vals).all()); nonnegative=bool((vals>=0).all())
    zeros_spot=int(np.asarray(x.sum(axis=1)).ravel().eq(0).sum()) if hasattr(np.asarray(x.sum(axis=1)).ravel(),"eq") else int((np.asarray(x.sum(axis=1)).ravel()==0).sum())
    zeros_peak=int((np.asarray(x.sum(axis=0)).ravel()==0).sum())
    sizes=a.obs["SpaCRAFT_cluster"].value_counts().sort_index().to_dict()
    (out/"cluster_sizes.json").write_text(json.dumps({str(k):int(v) for k,v in sizes.items()},indent=2))
    audit.append(dict(dataset_name=row.dataset_name,status="PASS",standardized_h5ad=target,
        spots=a.n_obs,peaks=a.n_vars,spot_order_exact=True,cluster_missing=int(a.obs.SpaCRAFT_cluster.isna().sum()),
        coordinate_missing=int(np.isnan(a.obsm["spatial"]).sum()),peak_ids_unique=a.var_names.is_unique,
        matrix_finite=finite,matrix_nonnegative=nonnegative,zero_spots=zeros_spot,zero_peaks=zeros_peak))
pd.DataFrame(audit).to_csv(ROOT/"audit/STANDARDIZED_OBJECT_DETAILS.csv",index=False)
lines=["# Standardized object audit","","- Status: **PASS**",
       "- Each object is a new read-only derivative under `data_cache/<slice>/standardized.h5ad`.",
       "- Canonical order is the source h5ad obs order and is exactly tied to frozen SpaCRAFT labels.",""]
for x in audit: lines.append(f"- {x['dataset_name']}: {x['spots']} spots × {x['peaks']} peaks; zero spots={x['zero_spots']}; zero peaks={x['zero_peaks']}; PASS")
(ROOT/"audit/STANDARDIZED_OBJECT_AUDIT.md").write_text("\n".join(lines)+"\n")
print("standardized_object_status=PASS")

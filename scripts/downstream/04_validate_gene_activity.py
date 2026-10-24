#!/usr/bin/env python3
"""Mechanical and policy validation for Phase 4."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json, re
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/gene_activity_phase4_v1"
MAT=OUT/"matrices"; META=OUT/"metadata"; FIG=ROOT/"figures/candidates/gene_activity_phase4_v1"
raw=ad.read_h5ad(MAT/"six_slice_gene_activity_raw.h5ad"); norm=ad.read_h5ad(MAT/"six_slice_gene_activity_normalized.h5ad")
checks={}
checks["spot_count_complete"]=raw.n_obs==norm.n_obs==17787
checks["same_sparse_dimensions"]=raw.shape==norm.shape and sparse.issparse(raw.X) and sparse.issparse(norm.X)
checks["canonical_order"]=np.array_equal(raw.obs.canonical_index.to_numpy(),np.arange(17787))
checks["required_obs"]=set(["canonical_index","section","stage","frozen_cluster"]).issubset(raw.obs)
checks["required_var"]=set(["gene_id","gene_symbol","chromosome","start","end","strand","peaks_contributing","scoring_rule"]).issubset(raw.var)
checks["gene_id_unique"]=raw.var_names.is_unique
checks["finite_nonnegative"]=all(np.isfinite(x.X.data).all() and (x.X.data>=0).all() for x in (raw,norm))
checks["nonzero_spots_ge_90pct"]=float(np.mean(np.asarray(raw.X.sum(1)).ravel()>0))>=.9
checks["clusters_present"]=set([0,5,6]).issubset(set(raw.obs.frozen_cluster.astype(int)))
checks["stage_complete"]=not raw.obs.stage.isna().any()
checks["pseudobulk_outputs"]=all((OUT/f"pseudobulk/C{c}_gene_activity_by_slice.csv.gz").exists() for c in (0,5,6))
checks["same_rules"]=raw.var.scoring_rule.nunique()==1
checks["axis_sensitivity_reported"]=(OUT/"sensitivity/GENE_ACTIVITY_PEAK_AXIS_SENSITIVITY.md").exists()
spfiles=list((OUT/"spatial_candidates").glob("*_spatial_plot_data.csv.gz"))
checks["spatial_global_scale"]=len(spfiles)==8 and all((lambda x: x.vmin.nunique()==1 and x.vmax.nunique()==1)(pd.read_csv(f)) for f in spfiles)
checks["figures_readable"]=all(f.stat().st_size>1000 for f in FIG.rglob("*.png"))
names="\n".join(str(p) for p in OUT.rglob("*")).lower()
checks["no_pvalue_fdr_outputs"]=not re.search(r"(^|[/_])(pvalue|fdr)([/_.]|$)",names)
checks["motif_go_not_run"]=not any(p.is_dir() and p.name.lower() in ("motif","go","enrichment") for p in OUT.rglob("*"))
checks["no_ground_truth_in_matrix"]=not any("ground" in c.lower() or "cell_type" in c.lower() for c in raw.obs)
checks["final_figure_not_assembled"]=not (ROOT/"figures/final/Figure6.png").exists()
report={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
 "explicit_policy":{"formal_inference_performed":False,"pvalues_reported":False,"spot_as_replicate_used":False,
 "models_retrained":False,"embedding_recomputed":False,"clusters_recomputed":False,"formal_inputs_modified":False,
 "pixel_level_registration_claimed":False}}
(META/"phase4_validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
assert report["status"]=="PASS",report
print(json.dumps(report,indent=2))

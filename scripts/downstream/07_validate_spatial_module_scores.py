#!/usr/bin/env python3
"""Mechanical and policy validation for Phase 7."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd
import anndata as ad
from PIL import Image

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/module_spatial_phase7_v1"
SCO=OUT/"scores"; META=OUT/"metadata"; MOD=ROOT/"results/accessibility_phase3_v1/modules"
x=pd.read_csv(SCO/"C5_spot_module_scores.csv.gz"); h=ad.read_h5ad(SCO/"C5_spot_module_scores.h5ad")
e=pd.read_csv(MOD/"C5_Module_EarlyHigh_peaks.csv"); l=pd.read_csv(MOD/"C5_Module_LateHigh_peaks.csv")
bm=pd.read_csv(OUT/"background_matching/background_match_summary.csv"); limits=json.loads((META/"spatial_global_color_limits.json").read_text())
checks={}
checks["module_peak_counts"]=len(e)==126 and len(l)==106
checks["spot_count_and_C5"]=len(x)==h.n_obs==17787 and int(x.is_C5.sum())==3208
checks["required_columns"]=set(["canonical_index","slice","stage","frozen_cluster","spatial_x","spatial_y","library_size","EarlyHigh_module_score","LateHigh_module_score","Late_minus_Early_score","EarlyHigh_unadjusted_score","LateHigh_unadjusted_score","is_C5"])<=set(x.columns)
checks["finite_scores"]=np.isfinite(x.select_dtypes("number")).all().all()
checks["canonical_order"]=np.array_equal(x.canonical_index,np.arange(17787))
checks["spatial_h5ad"]=h.obsm["spatial"].shape==(17787,2) and np.allclose(h.obsm["spatial"],x[["spatial_x","spatial_y"]])
checks["background_match_complete"]=len(bm)==2 and bm.match_success_fraction.eq(1).all()
checks["global_color_limits"]=all(np.isfinite(v).all() and v[0]<v[1] for v in limits.values())
checks["QC_outputs"]=all((OUT/p).exists() for p in ["QC/module_score_depth_correlation.csv","QC/module_score_method_concordance.csv","QC/module_score_distribution_by_slice.csv","sensitivity/MODULE_SCORE_METHOD_SENSITIVITY.md"])
checks["stage_summaries"]=all((OUT/p).exists() for p in ["stage_summary/C5_module_score_by_slice.csv","stage_summary/C5_module_score_by_stage_descriptive.csv"])
figs=list((ROOT/"figures/candidates/module_spatial_phase7_v1").glob("*.png"))
checks["candidate_figures_readable"]=len(figs)==4 and all((lambda p:p.stat().st_size>1000 and Image.open(p).verify() is None)(p) for p in figs)
checks["four_slice_audit_only"]=(META/"FOUR_SLICE_MODULE_PROJECTION_FEASIBILITY.md").exists() and not (OUT/"scores/four_slice_module_scores.csv").exists()
checks["final_figure_not_assembled"]=not (ROOT/"figures/final/Figure6.png").exists()
report={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
 "policy":{"formal_inference_performed":False,"spot_as_replicate_used":False,"models_retrained":False,
 "embedding_recomputed":False,"clusters_recomputed":False,"formal_inputs_modified":False,
 "gene_activity_run":False,"motif_run":False,"GO_run":False,"four_slice_projection_run":False}}
(META/"phase7_validation_report.json").write_text(json.dumps(report,indent=2,default=lambda z:bool(z))+"\n")
assert report["status"]=="PASS",report
print(json.dumps(report,indent=2,default=lambda z:bool(z)))

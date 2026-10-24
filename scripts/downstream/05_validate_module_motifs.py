#!/usr/bin/env python3
"""Validate Phase-5 inputs, matrices, results, figures, and policy boundaries."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import hashlib, json
import numpy as np
import pandas as pd
from scipy import sparse
from PIL import Image

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/motif_phase5_v1"
P3=ROOT/"results/accessibility_phase3_v1/downstream_inputs"; META=OUT/"metadata"; MM=OUT/"motif_matrix"
e=pd.read_csv(P3/"C5_EarlyHigh_peaks_for_motif.csv"); l=pd.read_csv(P3/"C5_LateHigh_peaks_for_motif.csv"); u=pd.read_csv(P3/"C5_tested_peak_background_for_motif.csv")
presence=sparse.load_npz(MM/"C5_tested_peak_motif_presence_sparse.npz"); scores=sparse.load_npz(MM/"C5_tested_peak_motif_scores_sparse.npz")
motifs=pd.read_csv(MM/"motif_metadata.csv"); cov=pd.read_csv(OUT/"sequence_audit/C5_tested_peak_sequence_covariates.csv.gz")
checks={}
checks["frozen_counts"]=len(e)==126 and len(l)==106 and len(u)==6540
checks["foreground_subsets_disjoint"]=set(e.peak_id)<=set(u.peak_id) and set(l.peak_id)<=set(u.peak_id) and not set(e.peak_id)&set(l.peak_id)
checks["unique_valid_peak_ids"]=all(x.peak_id.is_unique and x.peak_id.str.fullmatch(r"chr[^-]+-\d+-\d+").all() for x in (e,l,u))
checks["sequence_covariates_complete"]=len(cov)==6540 and np.isfinite(cov[["peak_length","GC_fraction","N_fraction","mean_accessibility"]]).all().all() and cov.N_fraction.max()<=.05
checks["sparse_matrices"]=sparse.issparse(presence) and sparse.issparse(scores) and presence.shape==scores.shape==(6540,879)
checks["score_support_matches_presence"]=np.array_equal(presence.indptr,scores.indptr) and np.array_equal(presence.indices,scores.indices)
checks["best_scores_finite"]=np.isfinite(scores.data).all()
checks["motif_count"]=len(motifs)==879 and motifs.motif_id.is_unique
params=json.loads((MM/"MOTIF_SCAN_PARAMETERS.json").read_text())
checks["resource_versions"]=params["genome_build"]=="mm10/GRCm38" and params["fasta_sha256"]=="db16cb4633191754f1d9cc70e73d2a1f60d03fdf62bcf4902a31a4717a3d2de7" and params["motif_database_sha256"]=="dd494278d356a4e170908c74d6d8eb746a2c7504d6210cd51017141f21a65b18"
checks["background_definitions"]=all(json.loads((OUT/m/"metadata.json").read_text())["background_total"]==6540-(126 if m=="EarlyHigh" else 106) for m in ("EarlyHigh","LateHigh"))
finite_cols=["corrected_odds_ratio","log2_corrected_odds_ratio","finite_adjusted_pvalue","minus_log10_FDR","logistic_motif_coefficient"]
checks["finite_formal_results"]=all(np.isfinite(pd.read_csv(OUT/m/"full_motif_enrichment_finite.csv")[finite_cols]).all().all() for m in ("EarlyHigh","LateHigh"))
checks["valid_fdr"]=all(pd.read_csv(OUT/m/"full_motif_enrichment_finite.csv").finite_adjusted_pvalue.between(0,1).all() for m in ("EarlyHigh","LateHigh"))
checks["bias_control_recorded"]=all("GC" in json.loads((OUT/m/"metadata.json").read_text())["bias_control"] for m in ("EarlyHigh","LateHigh"))
checks["sensitivity_reported"]=(OUT/"sensitivity/MOTIF_BACKGROUND_SENSITIVITY.md").exists()
figs=list((ROOT/"figures/candidates/motif_phase5_v1").glob("*.png"))
checks["candidate_figures_readable"]=len(figs)==4 and all((lambda p: p.stat().st_size>1000 and Image.open(p).verify() is None)(p) for p in figs)
checks["no_chromvar"]=not any("chromvar" in p.name.lower() for p in OUT.rglob("*"))
checks["go_not_run"]=not any(p.is_dir() and p.name.lower().startswith("go") for p in OUT.rglob("*"))
checks["final_figure_not_assembled"]=not (ROOT/"figures/final/Figure6.png").exists()
report={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
 "policy":{"formal_replicate_inference_performed":False,"spot_as_replicate_used":False,"models_retrained":False,
 "embedding_recomputed":False,"clusters_recomputed":False,"formal_inputs_modified":False,"chromVAR_run":False,"GO_run":False}}
(META/"phase5_validation_report.json").write_text(json.dumps(report,indent=2,default=lambda x:bool(x))+"\n")
assert report["status"]=="PASS",report
print(json.dumps(report,indent=2,default=lambda x:bool(x)))

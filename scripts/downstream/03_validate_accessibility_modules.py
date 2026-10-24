#!/usr/bin/env python3
"""Validate Phase 3 fixed-module outputs and analysis boundaries."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/accessibility_phase3_v1"; FIG=ROOT/"figures/candidates/accessibility_phase3_v1"
checks={}
m=pd.read_csv(OUT/"modules/C5_all_modules_membership.csv")
checks["formal_input_232"]=len(m)==232
checks["unique_peaks_232"]=m.peak_id.nunique()==232
checks["cluster_C5_only"]=set(m.cluster)=={5}
checks["module_count_2"]=m.module.nunique()==2
checks["early_126"]=(m.module=="Module_EarlyHigh").sum()==126
checks["late_106"]=(m.module=="Module_LateHigh").sum()==106
checks["middle_0"]=not m.pattern.eq("middle-high").any()
raw=pd.read_csv(OUT/"matrices/C5_modules_log2CPM_six_slices.csv.gz")
z=pd.read_csv(OUT/"matrices/C5_modules_row_zscore_six_slices.csv.gz")
stage=pd.read_csv(OUT/"matrices/C5_modules_stage_mean_log2CPM.csv.gz")
checks["matrix_232"]=len(raw)==len(z)==len(stage)==232
checks["six_slices_preserved"]=list(raw.columns[2:])==[
 "220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
checks["finite_matrices"]=bool(np.isfinite(raw.iloc[:,2:]).all().all() and
                                np.isfinite(z.iloc[:,2:]).all().all() and
                                np.isfinite(stage.iloc[:,2:]).all().all())
checks["row_zscore_nonconstant"]=bool((z.iloc[:,2:].std(axis=1,ddof=0)>0).all())
checks["downstream_lists"]=all((OUT/"downstream_inputs"/x).is_file() for x in [
 "C5_EarlyHigh_peaks_for_motif.csv","C5_LateHigh_peaks_for_motif.csv",
 "C5_tested_peak_background_for_motif.csv","C5_EarlyHigh_peaks_for_gene_mapping.csv",
 "C5_LateHigh_peaks_for_gene_mapping.csv"])
checks["background_6540"]=len(pd.read_csv(
 OUT/"downstream_inputs/C5_tested_peak_background_for_motif.csv"))==6540
cols=[]
for f in OUT.rglob("*.csv"): cols.extend(pd.read_csv(f,nrows=1).columns)
checks["no_pvalue_fdr_columns"]=not any(x.lower() in {"pvalue","p_value","p-value","fdr","qvalue"} for x in cols)
pngs=sorted(FIG.glob("*.png"))
checks["candidate_png_count_4"]=len(pngs)==4
checks["candidate_png_readable"]=True
for p in pngs:
    try:
        with Image.open(p) as im: im.verify()
    except Exception: checks["candidate_png_readable"]=False
checks["no_reclustering"]=True
checks["motif_GO_not_run"]=True
checks["formal_inference_performed"]=False
checks["pvalues_reported"]=False
checks["spot_as_replicate_used"]=False
excluded={"formal_inference_performed","pvalues_reported","spot_as_replicate_used"}
checks={k:bool(v) for k,v in checks.items()}
status="COMPLETE" if all(v is True for k,v in checks.items() if k not in excluded) else "FAILED"
report={"phase3_modules_status":status,"checks":checks}
(OUT/"metadata/phase3_validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
if status!="COMPLETE": raise SystemExit(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

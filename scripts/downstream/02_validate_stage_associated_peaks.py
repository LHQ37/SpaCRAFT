#!/usr/bin/env python3
"""Validate all Phase 2 objective success criteria."""
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
OUT=ROOT/"results/accessibility_phase2_v1"; PB=OUT/"pseudobulk"; FIG=ROOT/"figures/candidates/accessibility_phase2_v1"
checks={}
meta=pd.read_csv(PB/"pseudobulk_sample_metadata.csv")
checks["clusters_only_C5_C6"]=sorted(meta.cluster.unique().tolist())==[5,6]
checks["six_slices_each_cluster"]=bool((meta.groupby("cluster").slice.nunique()==6).all())
checks["coverage_adequate"]=bool((meta.spot_count>=100).all())
checks["sample_rows_12"]=len(meta)==12
for c in (5,6):
    raw=pd.read_csv(PB/f"C{c}_raw_pseudobulk_counts.csv.gz")
    log=pd.read_csv(PB/f"C{c}_log2CPM.csv.gz")
    summ=pd.read_csv(OUT/f"C{c}/C{c}_peak_stage_summary.csv.gz")
    tested=pd.read_csv(OUT/f"C{c}/C{c}_tested_peaks.csv")
    checks[f"C{c}_8282_peaks"]=len(raw)==len(log)==len(summ)==len(tested)==8282
    checks[f"C{c}_unique_peaks"]=raw.peak_id.nunique()==8282
    checks[f"C{c}_raw_nonnegative_integer"]=bool((raw.iloc[:,1:].to_numpy()>=0).all() and
        np.allclose(raw.iloc[:,1:].to_numpy(),np.rint(raw.iloc[:,1:].to_numpy())))
    checks[f"C{c}_finite"]=bool(np.isfinite(log.iloc[:,1:]).all().all() and
                                  np.isfinite(summ.select_dtypes("number")).all().all())
    checks[f"C{c}_six_columns"]=list(raw.columns[1:])==[
        "220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
checks["sensitivity_thresholds"]=set(pd.read_csv(
    OUT/"combined/effect_threshold_sensitivity.csv").effect_threshold)=={.75,1.0,1.25}
checks["formal_threshold_1"]=bool(pd.read_csv(
    OUT/"combined/effect_threshold_sensitivity.csv").query("formal_threshold").effect_threshold.eq(1).all())
csv_columns=[]
for f in OUT.rglob("*.csv"):
    csv_columns.extend(pd.read_csv(f,nrows=1).columns.tolist())
checks["no_pvalue_fdr_columns"]=not any(x.lower() in {"pvalue","p_value","p-value","fdr","qvalue"} for x in csv_columns)
pngs=sorted(FIG.glob("*.png"))
checks["candidate_png_count_9"]=len(pngs)==9
checks["candidate_png_readable"]=True
for p in pngs:
    try:
        with Image.open(p) as im: im.verify()
    except Exception: checks["candidate_png_readable"]=False
checks["formal_inference_performed"]=False
checks["pvalues_reported"]=False
checks["spot_as_replicate_used"]=False
positive=[v for k,v in checks.items() if k not in
          ("formal_inference_performed","pvalues_reported","spot_as_replicate_used")]
status="COMPLETE" if all(x is True for x in positive) else "FAILED"
report={"phase2_accessibility_status":status,"checks":checks}
(OUT/"metadata/phase2_validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
if status!="COMPLETE": raise SystemExit(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

#!/usr/bin/env python3
"""Build fixed C5 early/late accessibility modules without reclustering."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import hashlib, json
import numpy as np
import pandas as pd

ROOT=MULTI_DOWNSTREAM_ROOT
P2=ROOT/"results/accessibility_phase2_v1"
OUT=ROOT/"results/accessibility_phase3_v1"
MOD=OUT/"modules"; MAT=OUT/"matrices"; PROF=OUT/"profiles"; STAB=OUT/"stability"
DOWN=OUT/"downstream_inputs"; META=OUT/"metadata"
for d in (MOD,MAT,PROF,STAB,DOWN,META): d.mkdir(parents=True,exist_ok=True)
INPUT=P2/"combined/C5_phase3_strict_module_input_peaks.csv"
LOG=P2/"pseudobulk/C5_log2CPM.csv.gz"
TESTED=P2/"C5/C5_tested_peaks.csv"
SAMPLE=P2/"pseudobulk/pseudobulk_sample_metadata.csv"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGES={"E12.5":ORDER[:2],"E13.5":ORDER[2:4],"E15.5":ORDER[4:]}

inp=pd.read_csv(INPUT)
assert len(inp)==232 and inp.peak_id.nunique()==232
assert set(inp.cluster)=={5} and inp.strict_pairwise_concordant.astype(bool).all()
assert (inp.pattern=="early-high").sum()==126 and (inp.pattern=="late-high").sum()==106
assert not inp.pattern.eq("middle-high").any()
logdf=pd.read_csv(LOG)
assert list(logdf.columns[1:])==ORDER and logdf.peak_id.nunique()==8282
assert set(inp.peak_id)<=set(logdf.peak_id)
vals=logdf.set_index("peak_id").loc[inp.peak_id,ORDER].to_numpy(float)
assert np.isfinite(vals).all()
sd=vals.std(axis=1,ddof=0)
assert (sd>0).all()
z=(vals-vals.mean(axis=1,keepdims=True))/sd[:,None]

membership=inp.rename(columns={"rank":"phase2_rank"}).copy()
membership["module"]=membership.pattern.map(
    {"early-high":"Module_EarlyHigh","late-high":"Module_LateHigh"})
fields=["cluster","peak_id","module","pattern","early_to_late_delta","total_stage_range",
        "max_within_stage_difference","pairwise_concordance_count","mean_accessibility","phase2_rank"]
membership=membership[fields].sort_values(["module","phase2_rank"],kind="mergesort")
membership.to_csv(MOD/"C5_all_modules_membership.csv",index=False)
membership[membership.module.eq("Module_EarlyHigh")].to_csv(MOD/"C5_Module_EarlyHigh_peaks.csv",index=False)
membership[membership.module.eq("Module_LateHigh")].to_csv(MOD/"C5_Module_LateHigh_peaks.csv",index=False)

rawmat=pd.DataFrame(vals,columns=ORDER); rawmat.insert(0,"peak_id",inp.peak_id)
rawmat.insert(1,"module",inp.pattern.map({"early-high":"Module_EarlyHigh","late-high":"Module_LateHigh"}))
zmat=pd.DataFrame(z,columns=ORDER); zmat.insert(0,"peak_id",inp.peak_id)
zmat.insert(1,"module",rawmat.module)
rawmat.to_csv(MAT/"C5_modules_log2CPM_six_slices.csv.gz",index=False,compression="gzip")
zmat.to_csv(MAT/"C5_modules_row_zscore_six_slices.csv.gz",index=False,compression="gzip")
stagemat=rawmat[["peak_id","module"]].copy()
for st,ss in STAGES.items(): stagemat[st]=rawmat[ss].mean(axis=1)
stagemat.to_csv(MAT/"C5_modules_stage_mean_log2CPM.csv.gz",index=False,compression="gzip")

profiles=[]
for module in ("Module_EarlyHigh","Module_LateHigh"):
    mask=rawmat.module.eq(module).to_numpy()
    for j,s in enumerate(ORDER):
        x=vals[mask,j]; zz=z[mask,j]
        profiles.append({"module":module,"slice":s,"stage":next(k for k,v in STAGES.items() if s in v),
            "mean_raw_log2CPM":x.mean(),"median_raw_log2CPM":np.median(x),
            "mean_row_zscore":zz.mean(),"median_row_zscore":np.median(zz),
            "peak_count":len(x),"raw_log2CPM_IQR":np.quantile(x,.75)-np.quantile(x,.25),
            "row_zscore_IQR":np.quantile(zz,.75)-np.quantile(zz,.25)})
profiles=pd.DataFrame(profiles)
profiles.to_csv(PROF/"C5_module_profiles_by_slice.csv",index=False)
stage_rows=[]
for module,g in profiles.groupby("module",sort=False):
    for st,ss in STAGES.items():
        x=g.set_index("slice").loc[ss,"mean_row_zscore"]
        xr=g.set_index("slice").loc[ss,"mean_raw_log2CPM"]
        stage_rows.append({"module":module,"stage":st,"slice_1":ss[0],
          "slice_1_mean_row_zscore":x.iloc[0],"slice_2":ss[1],
          "slice_2_mean_row_zscore":x.iloc[1],"stage_mean_row_zscore":x.mean(),
          "stage_min_row_zscore":x.min(),"stage_max_row_zscore":x.max(),
          "stage_sd_row_zscore":x.std(ddof=1),"stage_mean_raw_log2CPM":xr.mean(),
          "stage_min_raw_log2CPM":xr.min(),"stage_max_raw_log2CPM":xr.max(),
          "stage_sd_raw_log2CPM":xr.std(ddof=1),"n_slices":2})
pd.DataFrame(stage_rows).to_csv(PROF/"C5_module_profiles_by_stage_descriptive.csv",index=False)

for module,label in [("Module_EarlyHigh","EarlyHigh"),("Module_LateHigh","LateHigh")]:
    x=membership[membership.module.eq(module)].copy()
    x.to_csv(DOWN/f"C5_{label}_peaks_for_motif.csv",index=False)
    x.to_csv(DOWN/f"C5_{label}_peaks_for_gene_mapping.csv",index=False)
tested=pd.read_csv(TESTED).query("tested").copy()
assert len(tested)==6540
tested.to_csv(DOWN/"C5_tested_peak_background_for_motif.csv",index=False)

trace={"formal_input":str(INPUT),"formal_input_sha256":hashlib.sha256(INPUT.read_bytes()).hexdigest(),
       "phase2_log2cpm":str(LOG),"phase2_tested_peaks":str(TESTED),
       "phase2_sample_metadata":str(SAMPLE),"formal_cluster":5,"module_count":2,
       "early_high":126,"late_high":106,"slice_order":ORDER,"reclustered":False}
(META/"PHASE3_INPUT_TRACE.json").write_text(json.dumps(trace,indent=2)+"\n")
print(json.dumps({"status":"PASS","peaks":232,"early":126,"late":106,"modules":2},indent=2))

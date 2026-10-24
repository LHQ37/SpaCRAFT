#!/usr/bin/env python3
"""Audit direction, same-stage consistency, and leave-one-slice-out stability."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/accessibility_phase3_v1"; MAT=OUT/"matrices"; PROF=OUT/"profiles"
STAB=OUT/"stability"; META=OUT/"metadata"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGES={"E12.5":ORDER[:2],"E13.5":ORDER[2:4],"E15.5":ORDER[4:]}

raw=pd.read_csv(MAT/"C5_modules_log2CPM_six_slices.csv.gz")
zdf=pd.read_csv(MAT/"C5_modules_row_zscore_six_slices.csv.gz")
prof=pd.read_csv(PROF/"C5_module_profiles_by_slice.csv")
membership=pd.read_csv(OUT/"modules/C5_all_modules_membership.csv")
assert len(raw)==len(zdf)==len(membership)==232

def corr(a,b): return float(np.corrcoef(a,b)[0,1])
def rankcorr(a,b):
    return corr(pd.Series(a).rank(method="average").to_numpy(),
                pd.Series(b).rank(method="average").to_numpy())
def expected(module,early,late):
    return early>late if module=="Module_EarlyHigh" else late>early
def e13_behavior(early,mid1,mid2,late):
    midpoint=(early+late)/2
    if (mid1-midpoint)*(mid2-midpoint)<0: return "SLICE_DISCORDANT"
    lo,hi=sorted([early,late]); q1=lo+.25*(hi-lo); q3=lo+.75*(hi-lo)
    mm=(mid1+mid2)/2
    if q1<=mm<=q3: return "INTERMEDIATE"
    return "EARLY_LIKE" if abs(mm-early)<abs(mm-late) else "LATE_LIKE"

stage_audit=[]; same=[]; loo=[]
for module in ("Module_EarlyHigh","Module_LateHigh"):
    r=raw[raw.module.eq(module)].set_index("peak_id")[ORDER]
    zz=zdf[zdf.module.eq(module)].set_index("peak_id")[ORDER]
    scores=zz.mean(axis=0)
    early=scores[ORDER[:2]].mean(); late=scores[ORDER[4:]].mean()
    behavior=e13_behavior(early,scores[ORDER[2]],scores[ORDER[3]],late)
    all_pair=(r[ORDER[:2]].min(axis=1)>r[ORDER[4:]].max(axis=1)).all() if module=="Module_EarlyHigh" else (
             r[ORDER[4:]].min(axis=1)>r[ORDER[:2]].max(axis=1)).all()
    stage_audit.append({"module":module,"E12.5_mean_row_zscore":early,
        "E13.5_slice_1_row_zscore":scores[ORDER[2]],"E13.5_slice_2_row_zscore":scores[ORDER[3]],
        "E13.5_mean_row_zscore":scores[ORDER[2:4]].mean(),"E15.5_mean_row_zscore":late,
        "expected_endpoint_direction":expected(module,early,late),
        "all_peaks_retain_phase2_all_four_pairwise_direction":bool(all_pair),
        "E13.5_behavior":behavior})
    for st,ss in STAGES.items():
        a=r[ss[0]].to_numpy(); b=r[ss[1]].to_numpy()
        d=abs(scores[ss[0]]-scores[ss[1]])
        same.append({"module":module,"stage":st,"slice_1":ss[0],"slice_2":ss[1],
            "slice_1_module_mean_row_zscore":scores[ss[0]],
            "slice_2_module_mean_row_zscore":scores[ss[1]],
            "module_score_difference":scores[ss[1]]-scores[ss[0]],
            "absolute_module_score_difference":d,"peak_profile_pearson":corr(a,b),
            "peak_profile_spearman":rankcorr(a,b),
            "interpretation":"descriptive only; not biological replicate validation"})
    full_centroid=r.mean(axis=1).to_numpy()
    for omitted in ORDER:
        remain=[x for x in ORDER if x!=omitted]
        ec=[x for x in ORDER[:2] if x!=omitted]
        lc=[x for x in ORDER[4:] if x!=omitted]
        e=float(r[ec].to_numpy().mean()); l=float(r[lc].to_numpy().mean())
        remain_centroid=r[remain].mean(axis=1).to_numpy()
        loo.append({"module":module,"omitted_slice":omitted,
            "remaining_E12.5_centroid":e,"remaining_E15.5_centroid":l,
            "expected_direction_retained":bool(expected(module,e,l)),
            "remaining_endpoint_difference_late_minus_early":l-e,
            "full_vs_remaining_peak_centroid_pearson":corr(full_centroid,remain_centroid),
            "full_vs_remaining_peak_centroid_spearman":rankcorr(full_centroid,remain_centroid)})
stage_audit=pd.DataFrame(stage_audit); same=pd.DataFrame(same); loo=pd.DataFrame(loo)
stage_audit.to_csv(PROF/"C5_module_stage_order_audit.csv",index=False)
same.to_csv(STAB/"C5_module_same_stage_consistency.csv",index=False)
loo.to_csv(STAB/"C5_module_leave_one_slice_out.csv",index=False)

quality=[]
for module in ("Module_EarlyHigh","Module_LateHigh"):
    n=int(loo[loo.module.eq(module)].expected_direction_retained.sum())
    maxdiff=float(same[same.module.eq(module)].absolute_module_score_difference.max())
    full=bool(stage_audit.set_index("module").loc[module,"expected_endpoint_direction"])
    stab=("STABLE_ALL_OMISSIONS" if n==6 else "STABLE_MOST_OMISSIONS" if n>=5
          else "SENSITIVE_TO_ONE_SLICE" if n==4 else "UNSTABLE")
    if not full or n<4: q="WEAK_MODULE"
    elif n==6 and maxdiff<=.5: q="STRONG_DESCRIPTIVE_MODULE"
    elif n>=5 and maxdiff<=.5: q="MODERATE_DESCRIPTIVE_MODULE"
    else: q="SLICE_SENSITIVE_MODULE"
    quality.append({"module":module,"expected_full_direction":full,
        "leave_one_out_direction_count":n,"leave_one_out_stability":stab,
        "max_same_stage_absolute_module_score_difference":maxdiff,
        "quality_status":q,
        "E13.5_behavior":stage_audit.set_index("module").loc[module,"E13.5_behavior"]})
quality=pd.DataFrame(quality)
quality.to_csv(META/"PHASE3_MODULE_QUALITY_RESULTS.csv",index=False)
summary={"quality":quality.to_dict("records"),
         "exploratory_submodules_status":"EXPLORATORY_SUBMODULES_SKIPPED",
         "formal_inference_performed":False,"pvalues_reported":False,"spot_as_replicate_used":False}
(META/"phase3_analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

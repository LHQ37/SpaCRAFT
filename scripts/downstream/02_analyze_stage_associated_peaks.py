#!/usr/bin/env python3
"""Normalize pseudobulks and identify preregistered descriptive stage patterns."""
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
OUT=ROOT/"results/accessibility_phase2_v1"; PB=OUT/"pseudobulk"; QC=OUT/"qc"
COMB=OUT/"combined"; META=OUT/"metadata"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGES={"E12.5":ORDER[0:2],"E13.5":ORDER[2:4],"E15.5":ORDER[4:6]}
STABILITY=1.0

def corr(a,b): return float(np.corrcoef(a,b)[0,1])
def classify_quality(cons, ranked):
    strict=ranked[ranked.strict_pairwise_concordant]
    ne=(strict.pattern=="early-high").sum(); nl=(strict.pattern=="late-high").sum()
    pears=cons.pearson_correlation.to_numpy()
    if ne>=10 and nl>=10 and np.all(pears>=.90): return "STRONG_DESCRIPTIVE_PATTERN"
    if len(ranked)==0: return "INSUFFICIENT_CANDIDATES"
    if np.median(pears)>=.70: return "MODERATE_DESCRIPTIVE_PATTERN"
    return "WEAK_INCONSISTENT_PATTERN"

all_results={}; sensitivity_sets={}
for cluster in (5,6):
    cdir=OUT/f"C{cluster}"
    rawdf=pd.read_csv(PB/f"C{cluster}_raw_pseudobulk_counts.csv.gz")
    detdf=pd.read_csv(QC/f"C{cluster}_spot_detection_fraction.csv.gz")
    peaks=rawdf.peak_id.astype(str); raw=rawdf[ORDER].to_numpy(dtype=float)
    det=detdf[ORDER].to_numpy(dtype=float)
    libs=raw.sum(axis=0)
    assert (libs>0).all()
    cpm=raw/libs*1e6
    log=np.log2(cpm+1.0)
    assert np.isfinite(log).all()
    norm=pd.DataFrame(log,columns=ORDER); norm.insert(0,"peak_id",peaks)
    norm.to_csv(PB/f"C{cluster}_log2CPM.csv.gz",index=False,compression="gzip")

    cpm_primary=(cpm>=1).sum(axis=1)>=2
    stage_half=np.zeros(len(peaks),dtype=bool)
    for ss in STAGES.values():
        jj=[ORDER.index(x) for x in ss]
        stage_half |= (cpm[:,jj]>=.5).all(axis=1)
    cpm_pass=cpm_primary|stage_half
    detection_pass=(det>=.05).any(axis=1)
    tested=cpm_pass&detection_pass
    reasons=np.where(tested,"PASS",
             np.where(~cpm_pass & ~detection_pass,"FAIL_CPM_AND_DETECTION",
             np.where(~cpm_pass,"FAIL_CPM","FAIL_DETECTION")))
    tdf=pd.DataFrame({"peak_id":peaks,"tested":tested,"cpm_rule_pass":cpm_pass,
                      "spot_detection_rule_pass":detection_pass,"filter_status":reasons,
                      "max_spot_detection_fraction":det.max(axis=1),
                      "pseudobulks_CPM_ge_1":(cpm>=1).sum(axis=1)})
    tdf.to_csv(cdir/f"C{cluster}_tested_peaks.csv",index=False)
    filter_summary={"cluster":cluster,"original_peak_count":len(peaks),
        "retained_peak_count":int(tested.sum()),"filtered_peak_count":int((~tested).sum()),
        "fail_cpm_only":int(((~cpm_pass)&detection_pass).sum()),
        "fail_detection_only":int((cpm_pass&(~detection_pass)).sum()),
        "fail_both":int(((~cpm_pass)&(~detection_pass)).sum())}
    (cdir/f"C{cluster}_low_information_filter_summary.json").write_text(
        json.dumps(filter_summary,indent=2)+"\n")

    means={st:log[:,[ORDER.index(x) for x in ss]].mean(axis=1) for st,ss in STAGES.items()}
    mins={st:log[:,[ORDER.index(x) for x in ss]].min(axis=1) for st,ss in STAGES.items()}
    maxs={st:log[:,[ORDER.index(x) for x in ss]].max(axis=1) for st,ss in STAGES.items()}
    wd=np.column_stack([np.abs(log[:,0]-log[:,1]),np.abs(log[:,2]-log[:,3]),
                        np.abs(log[:,4]-log[:,5])])
    delta=means["E15.5"]-means["E12.5"]
    smat=np.column_stack([means[x] for x in STAGES])
    total_range=smat.max(axis=1)-smat.min(axis=1)
    early=log[:,0:2]; middle=log[:,2:4]; late=log[:,4:6]
    late_pairs=(late[:,:,None]>early[:,None,:]).sum(axis=(1,2))
    early_pairs=(early[:,:,None]>late[:,None,:]).sum(axis=(1,2))
    stable=wd.max(axis=1)<=STABILITY

    summary=pd.DataFrame({"peak_id":peaks})
    for j,s in enumerate(ORDER): summary[f"{s}_log2CPM"]=log[:,j]
    for st in STAGES:
        summary[f"{st}_mean"]=means[st]; summary[f"{st}_min"]=mins[st]
        summary[f"{st}_max"]=maxs[st]; summary[f"{st}_range"]=maxs[st]-mins[st]
    summary["early_to_late_delta"]=delta
    summary["total_stage_range"]=total_range
    summary["mean_accessibility"]=log.mean(axis=1)
    summary["E12.5_within_stage_absolute_difference"]=wd[:,0]
    summary["E13.5_within_stage_absolute_difference"]=wd[:,1]
    summary["E15.5_within_stage_absolute_difference"]=wd[:,2]
    summary["max_within_stage_difference"]=wd.max(axis=1)
    summary["tested"]=tested
    summary["mean_monotonic_increasing"]=(smat[:,0]<=smat[:,1])&(smat[:,1]<=smat[:,2])
    summary["mean_monotonic_decreasing"]=(smat[:,0]>=smat[:,1])&(smat[:,1]>=smat[:,2])
    summary["strict_slice_supported_monotonic_increasing"]=(early.max(axis=1)<middle.min(axis=1))&(middle.max(axis=1)<late.min(axis=1))
    summary["strict_slice_supported_monotonic_decreasing"]=(early.min(axis=1)>middle.max(axis=1))&(middle.min(axis=1)>late.max(axis=1))

    middle_flag=tested&stable&(middle.min(axis=1)>means["E12.5"])&(
        middle.min(axis=1)>means["E15.5"])&((means["E13.5"]-means["E12.5"])>=1)&(
        (means["E13.5"]-means["E15.5"])>=1)
    summary["formal_middle_high"]=middle_flag

    cons=[]
    for st,ss in STAGES.items():
        a=log[tested,ORDER.index(ss[0])]; b=log[tested,ORDER.index(ss[1])]
        ra=pd.Series(a).rank(method="average").to_numpy()
        rb=pd.Series(b).rank(method="average").to_numpy()
        adiff=np.abs(a-b)
        cons.append({"cluster":cluster,"stage":st,"slice_1":ss[0],"slice_2":ss[1],
            "tested_peak_count":len(a),"pearson_correlation":corr(a,b),
            "spearman_correlation":corr(ra,rb),"median_absolute_difference":np.median(adiff),
            "p90_absolute_difference":np.quantile(adiff,.9),"peak_rank_correlation":corr(ra,rb),
            "interpretation":"descriptive only; not biological replicate validation"})
    cons=pd.DataFrame(cons)
    cons.to_csv(QC/f"C{cluster}_same_stage_peak_consistency.csv",index=False)
    summary[["peak_id","E12.5_within_stage_absolute_difference",
        "E13.5_within_stage_absolute_difference","E15.5_within_stage_absolute_difference",
        "max_within_stage_difference"]].to_csv(
        QC/f"C{cluster}_peak_level_within_stage_variability.csv.gz",index=False,compression="gzip")

    candidate_rows=[]
    sensitivity_sets[cluster]={}
    for threshold in (0.75,1.0,1.25):
        late_flag=tested&stable&(delta>=threshold)&(late.min(axis=1)>np.median(early,axis=1))&(late_pairs>=3)
        early_flag=tested&stable&(delta<=-threshold)&(early.min(axis=1)>np.median(late,axis=1))&(early_pairs>=3)
        sensitivity_sets[cluster][threshold]={
            "early":set(peaks[early_flag]),"late":set(peaks[late_flag]),
            "strict_early":set(peaks[early_flag&(early_pairs==4)]),
            "strict_late":set(peaks[late_flag&(late_pairs==4)])}
        if threshold==1.0:
            summary["formal_late_high"]=late_flag
            summary["formal_early_high"]=early_flag
            for pattern,flag,pairs in [("early-high",early_flag,early_pairs),("late-high",late_flag,late_pairs),
                                       ("middle-high",middle_flag,np.zeros(len(peaks),dtype=int))]:
                for i in np.where(flag)[0]:
                    strict=bool(pairs[i]==4) if pattern!="middle-high" else False
                    candidate_rows.append({"cluster":cluster,"peak_id":peaks.iloc[i],"pattern":pattern,
                        "early_to_late_delta":delta[i],"total_stage_range":total_range[i],
                        "max_within_stage_difference":wd[i].max(),
                        "pairwise_concordance_count":int(pairs[i]),
                        "strict_pairwise_concordant":strict,
                        "mean_accessibility":log[i].mean(),
                        "descriptive_confidence":("STRICT_PAIRWISE_DESCRIPTIVE" if strict else
                            "LOWER_CONFIDENCE_DESCRIPTIVE_PATTERN" if pattern=="middle-high"
                            else "DESCRIPTIVE_PATTERN")})
    summary.to_csv(cdir/f"C{cluster}_peak_stage_summary.csv.gz",index=False,compression="gzip")
    ranked=pd.DataFrame(candidate_rows)
    if len(ranked):
        ranked["_strict"]=ranked.strict_pairwise_concordant.astype(int)
        ranked["_absdelta"]=ranked.early_to_late_delta.abs()
        ranked=ranked.sort_values(["_strict","_absdelta","max_within_stage_difference",
                                  "mean_accessibility","peak_id"],
                                 ascending=[False,False,True,False,True],kind="mergesort")
        ranked["rank"]=np.arange(1,len(ranked)+1)
        ranked=ranked.drop(columns=["_strict","_absdelta"])
    else:
        ranked=pd.DataFrame(columns=["cluster","peak_id","pattern","early_to_late_delta",
          "total_stage_range","max_within_stage_difference","pairwise_concordance_count",
          "strict_pairwise_concordant","mean_accessibility","descriptive_confidence","rank"])
    ranked.to_csv(cdir/f"C{cluster}_ranked_stage_associated_peaks.csv",index=False)
    phase3=ranked[(ranked.pattern.isin(["early-high","late-high"]))&
                  ranked.strict_pairwise_concordant].copy()
    phase3.to_csv(COMB/f"C{cluster}_phase3_strict_module_input_peaks.csv",index=False)
    quality=classify_quality(cons,ranked)
    all_results[cluster]={"tested":int(tested.sum()),"filter":filter_summary,
        "early":int((summary.formal_early_high).sum()),"late":int((summary.formal_late_high).sum()),
        "middle":int(middle_flag.sum()),"strict":int(ranked.strict_pairwise_concordant.sum()),
        "quality":quality,"same_stage_pearson":dict(zip(cons.stage,cons.pearson_correlation))}

sens_rows=[]
for threshold in (0.75,1.0,1.25):
    for pattern in ("early","late"):
        a=sensitivity_sets[5][threshold][pattern]; b=sensitivity_sets[6][threshold][pattern]
        sa=sensitivity_sets[5][threshold]["strict_"+pattern]
        sb=sensitivity_sets[6][threshold]["strict_"+pattern]
        sens_rows.append({"effect_threshold":threshold,"pattern":pattern+"-high",
            "C5_count":len(a),"C6_count":len(b),"C5_C6_overlap_count":len(a&b),
            "C5_strict_pairwise_count":len(sa),"C6_strict_pairwise_count":len(sb),
            "strict_overlap_count":len(sa&sb),"formal_threshold":threshold==1.0})
pd.DataFrame(sens_rows).to_csv(COMB/"effect_threshold_sensitivity.csv",index=False)
(META/"phase2_analysis_summary.json").write_text(json.dumps(all_results,indent=2)+"\n")
print(json.dumps(all_results,indent=2))

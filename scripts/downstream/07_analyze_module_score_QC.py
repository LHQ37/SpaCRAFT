#!/usr/bin/env python3
"""Descriptive depth QC, stage summaries, and method sensitivity."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/module_spatial_phase7_v1"
SCO=OUT/"scores"; QC=OUT/"QC"; ST=OUT/"stage_summary"; SENS=OUT/"sensitivity"; META=OUT/"metadata"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
x=pd.read_csv(SCO/"C5_spot_module_scores.csv.gz"); c=x[x.is_C5.astype(bool)].copy(); assert len(c)==3208
corr=[]
for m in ("EarlyHigh","LateHigh"):
 score=f"{m}_module_score"
 for cov in ("library_size","detected_peak_count","slice_total_depth",f"{m}_background_score"):
  corr.append({"module":m,"score":"bias_controlled","covariate":cov,"pearson":pearsonr(c[score],c[cov]).statistic,
   "spearman":spearmanr(c[score],c[cov]).statistic,"spot_scope":"frozen_C5"})
pd.DataFrame(corr).to_csv(QC/"module_score_depth_correlation.csv",index=False)

method=[]; sens=[]
for m in ("EarlyHigh","LateHigh"):
 a=c[f"{m}_module_score"].to_numpy(); u=c[f"{m}_unadjusted_score"].to_numpy(); n=len(c); k=max(1,int(.1*n))
 topa=set(np.argsort(a)[-k:]); topu=set(np.argsort(u)[-k:]); bota=set(np.argsort(a)[:k]); botu=set(np.argsort(u)[:k])
 row={"module":m,"pearson":pearsonr(a,u).statistic,"spearman":spearmanr(a,u).statistic,
      "top_decile_overlap_fraction":len(topa&topu)/k,"bottom_decile_overlap_fraction":len(bota&botu)/k}
 method.append(row)
pd.DataFrame(method).to_csv(QC/"module_score_method_concordance.csv",index=False)

dist=[]
for m in ("EarlyHigh","LateHigh"):
 for s,z in c.groupby("slice"):
  v=z[f"{m}_module_score"]
  dist.append({"module":m,"slice":s,"stage":z.stage.iloc[0],"spot_count":len(z),"mean":v.mean(),"median":v.median(),
    "q25":v.quantile(.25),"q75":v.quantile(.75),"IQR":v.quantile(.75)-v.quantile(.25),
    "unadjusted_mean":z[f"{m}_unadjusted_score"].mean()})
d=pd.DataFrame(dist); d["slice_order"]=d.slice.map({s:i for i,s in enumerate(ORDER)}); d=d.sort_values(["module","slice_order"]).drop(columns="slice_order")
d.to_csv(QC/"module_score_distribution_by_slice.csv",index=False); d.to_csv(ST/"C5_module_score_by_slice.csv",index=False)
stage=(d.groupby(["module","stage"]).agg(stage_mean=("mean","mean"),slice_mean_min=("mean","min"),slice_mean_max=("mean","max"),
 stage_median_of_slice_medians=("median","median"),slice_count=("slice","size"),total_C5_spots=("spot_count","sum")).reset_index())
stage.to_csv(ST/"C5_module_score_by_stage_descriptive.csv",index=False)

def status_for(vals,module):
 e=vals[:2]; mid=vals[2:4]; l=vals[4:]
 direction=(min(e)>max(l)) if module=="EarlyHigh" else (min(l)>max(e))
 sm=[e.mean(),mid.mean(),l.mean()]; intermediate=min(sm[0],sm[2])<sm[1]<max(sm[0],sm[2])
 slice_intermediate=all(min(sm[0],sm[2])<v<max(sm[0],sm[2]) for v in mid)
 return direction,intermediate,slice_intermediate
stageflags={}; expected_all=True; intermediate_all=True; sliceinter_all=True
for m in ("EarlyHigh","LateHigh"):
 z=d[d.module.eq(m)].set_index("slice").loc[ORDER]; fl=status_for(z["mean"].to_numpy(),m)
 stageflags[m]={"expected_early_late_direction":bool(fl[0]),"E13_stage_mean_intermediate":bool(fl[1]),"both_E13_slice_means_intermediate":bool(fl[2])}
 expected_all &= fl[0]; intermediate_all &= fl[1]; sliceinter_all &= fl[2]

# Sensitivity includes spot correlation, expected slice direction, stage ordering, and extreme overlap.
for row in method:
 m=row["module"]; z=d[d.module.eq(m)].set_index("slice").loc[ORDER]
 unadj_dir=status_for(z.unadjusted_mean.to_numpy(),m)[0]; unadj_mid=status_for(z.unadjusted_mean.to_numpy(),m)[1]
 sens.append({**row,"adjusted_expected_slice_direction":stageflags[m]["expected_early_late_direction"],
  "unadjusted_expected_slice_direction":bool(unadj_dir),"adjusted_stage_ordering":stageflags[m]["E13_stage_mean_intermediate"],
  "unadjusted_stage_ordering":bool(unadj_mid)})
s=pd.DataFrame(sens)
high=(s.pearson.ge(.8)&s.spearman.ge(.8)&s.top_decile_overlap_fraction.ge(.5)&s.bottom_decile_overlap_fraction.ge(.5)&s.adjusted_expected_slice_direction&s.unadjusted_expected_slice_direction&s.adjusted_stage_ordering&s.unadjusted_stage_ordering).all()
moderate=(s.pearson.ge(.6)&s.spearman.ge(.6)&s.top_decile_overlap_fraction.ge(.25)&s.bottom_decile_overlap_fraction.ge(.25)&s.adjusted_expected_slice_direction&s.unadjusted_expected_slice_direction).all()
sens_status="HIGH_CONCORDANCE" if high else ("MODERATE_CONCORDANCE" if moderate else "LOW_CONCORDANCE")
s["method_sensitivity_status"]=sens_status; s.to_csv(SENS/"MODULE_SCORE_METHOD_SENSITIVITY.csv",index=False)
(SENS/"MODULE_SCORE_METHOD_SENSITIVITY.md").write_text("# Module-score method sensitivity\n\n"
 f"Status: **{sens_status}**\n\nBias-controlled scores were compared with the pre-registered unadjusted normalized module means among frozen C5 spots.\n\n"+
 "\n".join(f"- {z.module}: Pearson={z.pearson:.3f}, Spearman={z.spearman:.3f}, top/bottom decile overlap={z.top_decile_overlap_fraction:.3f}/{z.bottom_decile_overlap_fraction:.3f}." for z in s.itertuples())+"\n")

depth=pd.DataFrame(corr); dd=depth[depth.covariate.isin(["library_size","detected_peak_count"])]
mx=dd.groupby("module")[["pearson","spearman"]].apply(lambda z:np.abs(z.to_numpy()).max()).to_dict()
bgok=pd.read_csv(OUT/"background_matching/background_match_summary.csv").match_success_fraction.eq(1).all()
quality="HIGH_ROBUSTNESS" if bgok and max(mx.values())<=.2 else ("MODERATE_ROBUSTNESS" if bgok and max(mx.values())<=.35 else "DEPTH_SENSITIVE")
summary={"C5_spot_count":len(c),"module_score_quality_status":quality,"method_sensitivity_status":sens_status,
 "max_absolute_depth_correlation":mx,"stage_direction":stageflags,
 "E13_5_behavior":"INTERMEDIATE" if intermediate_all else "NOT_INTERMEDIATE",
 "same_stage_slice_consistency":"CONSISTENT" if sliceinter_all and expected_all else "INCONSISTENT"}
(META/"phase7_analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

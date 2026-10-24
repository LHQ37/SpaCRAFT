#!/usr/bin/env python3
"""Pre-registered descriptive gene-activity dynamics; no inferential statistics."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/gene_activity_phase4_v1"; PB=OUT/"pseudobulk"; SENS=OUT/"sensitivity"; META=OUT/"metadata"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGES=[("E12.5",0,1),("E13.5",2,3),("E15.5",4,5)]

def analyze(tab,cluster,threshold=.75):
    vals=tab[[f"{s}_log2CPM" for s in ORDER]].to_numpy(float)
    det=tab[[f"{s}_spot_detection_fraction" for s in ORDER]].to_numpy(float)
    tested=(np.sum(vals>=1,axis=1)>=2)&(det.max(1)>=.05)&(tab.peaks_contributing.to_numpy()>0)&tab.gene_symbol.fillna("").ne("").to_numpy()&(np.ptp(vals,axis=1)>0)
    em=vals[:,:2].mean(1); mm=vals[:,2:4].mean(1); lm=vals[:,4:].mean(1)
    within=np.max(np.c_[np.abs(vals[:,0]-vals[:,1]),np.abs(vals[:,2]-vals[:,3]),np.abs(vals[:,4]-vals[:,5])],axis=1)
    ep=(vals[:,:2].min(1)>vals[:,4:].max(1)); lp=(vals[:,4:].min(1)>vals[:,:2].max(1))
    delta=lm-em
    early=tested&ep&(delta<=-threshold)&(within<=1)
    late=tested&lp&(delta>=threshold)&(within<=1)
    middle=tested&(mm-em>=.75)&(mm-lm>=.75)&(within<=1)&~early&~late
    label=np.where(early,"EarlyHigh",np.where(late,"LateHigh",np.where(middle,"MiddleHigh","Other")))
    out=tab[["gene_id","gene_symbol","peaks_contributing"]].copy()
    for j,s in enumerate(ORDER): out[s]=vals[:,j]
    out["tested"]=tested; out["early_mean"]=em; out["middle_mean"]=mm; out["late_mean"]=lm
    for name,i,j in STAGES:
        out[f"{name}_mean"]=vals[:,[i,j]].mean(1); out[f"{name}_min"]=vals[:,[i,j]].min(1); out[f"{name}_max"]=vals[:,[i,j]].max(1)
    out["late_minus_early_delta"]=delta; out["early_to_late_delta"]=delta
    out["total_stage_range"]=np.ptp(np.c_[em,mm,lm],axis=1); out["max_within_stage_difference"]=within
    out["mean_activity"]=vals.mean(1); out["max_spot_detection_fraction"]=det.max(1)
    out["detection_prevalence"]=det.mean(1)
    out["pairwise_concordance_count"]=np.maximum((vals[:,:2,None]>vals[:,None,4:]).sum((1,2)),(vals[:,:2,None]<vals[:,None,4:]).sum((1,2)))
    out["all_four_pairwise_concordant"]=ep|lp; out["strict_pairwise_concordant"]=ep|lp; out["pattern"]=label
    out["descriptive_confidence"]=np.where(out.pattern.eq("MiddleHigh"),"LOWER_CONFIDENCE_DESCRIPTIVE_PATTERN",
                                  np.where(out.pattern.isin(["EarlyHigh","LateHigh"]),"STRICT_DESCRIPTIVE_PATTERN","NOT_CANDIDATE"))
    out["rank_score"]=np.nan
    for pat in ("EarlyHigh","LateHigh","MiddleHigh"):
        ix=out.pattern.eq(pat)
        x=out[ix].sort_values(["all_four_pairwise_concordant","late_minus_early_delta","max_within_stage_difference","mean_activity","max_spot_detection_fraction","gene_symbol"],
            ascending=[False,pat=="EarlyHigh",True,False,False,True])
        out.loc[x.index,"rank_score"]=np.arange(1,len(x)+1)
    cor=[]
    for st,i,j in STAGES:
        use=tested
        cor.append({"cluster":cluster,"stage":st,"slice_1":ORDER[i],"slice_2":ORDER[j],
                    "pearson":np.corrcoef(vals[use,i],vals[use,j])[0,1],
                    "spearman":spearmanr(vals[use,i],vals[use,j]).statistic,"tested_gene_count":int(use.sum())})
    return out,pd.DataFrame(cor)

summ=[]; cors=[]; allres={}
for c in (0,5,6):
    t=pd.read_csv(PB/f"C{c}_gene_activity_by_slice.csv.gz")
    r,co=analyze(t,c); allres[c]=r; cors.append(co)
    r.to_csv(OUT/f"C{c}"/f"C{c}_ranked_gene_activity_dynamics.csv",index=False)
    r[r.tested].to_csv(OUT/f"C{c}"/f"C{c}_tested_genes.csv",index=False)
    r[r.tested].to_csv(OUT/f"C{c}"/f"C{c}_gene_stage_summary.csv.gz",index=False,compression="gzip")
    ranked=r[r.pattern.ne("Other")].copy(); ranked.insert(0,"cluster",c); ranked["rank"]=ranked.rank_score
    ranked.to_csv(OUT/f"C{c}"/f"C{c}_ranked_stage_associated_genes.csv",index=False)
    summ.append({"cluster":c,"tested_gene_count":int(r.tested.sum()),"early_high_count":int((r.pattern=="EarlyHigh").sum()),
      "late_high_count":int((r.pattern=="LateHigh").sum()),"middle_high_count":int((r.pattern=="MiddleHigh").sum())})
pd.concat(cors).to_csv(META/"same_stage_gene_activity_consistency.csv",index=False)
pd.DataFrame(summ).to_csv(META/"cluster_gene_activity_summary.csv",index=False)

# Threshold sensitivity.
rows=[]
for c in (0,5,6):
    t=pd.read_csv(PB/f"C{c}_gene_activity_by_slice.csv.gz")
    for th in (.5,.75,1.0):
        r,_=analyze(t,c,th)
        rows.append({"cluster":c,"effect_threshold":th,"tested":int(r.tested.sum()),
                     "early_high":int((r.pattern=="EarlyHigh").sum()),"late_high":int((r.pattern=="LateHigh").sum())})
pd.DataFrame(rows).to_csv(SENS/"GENE_ACTIVITY_EFFECT_THRESHOLD_SENSITIVITY.csv",index=False)

# C5 fixed-candidate leave-one-slice-out direction audit.
r=allres[5]; fixed=r[r.pattern.isin(["EarlyHigh","LateHigh"])].copy(); loo=[]
vals=fixed[ORDER].to_numpy()
for k,s in enumerate(ORDER):
    e=[i for i in (0,1) if i!=k]; l=[i for i in (4,5) if i!=k]
    d=vals[:,l].mean(1)-vals[:,e].mean(1)
    ok=np.where(fixed.pattern.eq("EarlyHigh"),d<0,d>0)
    for gene,sym,pat,dd,keep in zip(fixed.gene_id,fixed.gene_symbol,fixed.pattern,d,ok):
        loo.append({"gene_id":gene,"gene_symbol":sym,"fixed_pattern":pat,"omitted_slice":s,
                    "leave_one_out_delta":dd,"direction_retained":bool(keep)})
loo=pd.DataFrame(loo); loo.to_csv(OUT/"C5"/"C5_gene_leave_one_slice_out.csv",index=False)
loo_rate=float(loo.direction_retained.mean()) if len(loo) else np.nan

# Primary-versus-8,282-axis sensitivity on C5.
st=pd.read_csv(SENS/"C5_8282_gene_activity_by_slice.csv.gz"); sr,_=analyze(st,5)
p=r[r.tested].merge(sr[sr.tested],on=["gene_id","gene_symbol"],suffixes=("_primary","_8282"))
pear=float(p.late_minus_early_delta_primary.corr(p.late_minus_early_delta_8282)) if len(p)>2 else np.nan
def tops(x,pat): return set(x[x.pattern==pat].sort_values("rank_score").gene_id.head(20))
ovs=[]
for pat in ("EarlyHigh","LateHigh"):
    a,b=tops(r,pat),tops(sr,pat); ovs.append(len(a&b)/max(1,min(20,len(a),len(b))))
matched=float((np.sign(p.late_minus_early_delta_primary)==np.sign(p.late_minus_early_delta_8282)).mean()) if len(p) else np.nan
ov=min(ovs)
axis="HIGH_CONCORDANCE" if pear>=.8 and ov>=.5 and matched>=.8 else ("MODERATE_CONCORDANCE" if pear>=.6 and ov>=.25 and matched>=.6 else "LOW_CONCORDANCE")
p.to_csv(SENS/"GENE_ACTIVITY_PEAK_AXIS_SENSITIVITY.csv",index=False)
(SENS/"GENE_ACTIVITY_PEAK_AXIS_SENSITIVITY.md").write_text(
 f"# Peak-axis sensitivity\n\nStatus: `{axis}`\n\nDelta Pearson: {pear:.4f}; minimum top-20 overlap fraction: {ov:.4f}; matched direction: {matched:.4f}.\n")

# Quality status per cluster (C5 includes fixed-candidate LOO; others are not penalized for an inapplicable LOO).
q=[]; co=pd.concat(cors)
for x in summ:
    c=x["cluster"]; med=float(co[co.cluster==c].pearson.median()); loo_c=loo_rate if c==5 else np.nan
    enough_s=x["early_high_count"]>=10 and x["late_high_count"]>=10 and med>=.8 and axis=="HIGH_CONCORDANCE" and (c!=5 or loo_c>=.9)
    enough_m=x["early_high_count"]>=3 and x["late_high_count"]>=3 and med>=.6 and axis!="LOW_CONCORDANCE" and (c!=5 or loo_c>=.8)
    status="STRONG_DESCRIPTIVE_GENE_PATTERN" if enough_s else ("MODERATE_DESCRIPTIVE_GENE_PATTERN" if enough_m else ("INSUFFICIENT_CANDIDATES" if x["early_high_count"]+x["late_high_count"]==0 else "WEAK_INCONSISTENT_GENE_PATTERN"))
    q.append({**x,"median_same_stage_pearson":med,"C5_leave_one_out_direction_fraction":loo_c,
              "peak_axis_sensitivity_status":axis,"quality_status":status})
pd.DataFrame(q).to_csv(META/"PHASE4_GENE_ACTIVITY_QUALITY_RESULTS.csv",index=False)
summary={"peak_axis_sensitivity_status":axis,"delta_pearson":pear,"top20_overlap_min":ov,
 "matched_direction":matched,"C5_leave_one_out_direction_fraction":loo_rate,"clusters":q}
(META/"phase4_analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

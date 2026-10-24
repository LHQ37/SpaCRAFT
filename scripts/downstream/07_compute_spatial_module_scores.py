#!/usr/bin/env python3
"""Sparse, library-normalized, covariate-matched module scores for all spots."""
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
from scipy import sparse
from scipy.spatial import cKDTree

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/module_spatial_phase7_v1"
SCO=OUT/"scores"; BM=OUT/"background_matching"; META=OUT/"metadata"
for d in (SCO,BM,META,OUT/"QC",OUT/"stage_summary",OUT/"sensitivity"): d.mkdir(parents=True,exist_ok=True)
H5=DATA_ROOT / "multi_slice_unzipped/多切片/Mouse_Embryo/mouse_embryo.h5ad"
OBS=MULTISLICE_RESULTS_ROOT / "canonical/six_slices_obs.csv"
LAB=MULTISLICE_RESULTS_ROOT / "six_slices/SpaCRAFT/cluster_labels.csv"
MOD=ROOT/"results/accessibility_phase3_v1/modules"; COV=ROOT/"results/motif_phase5_v1/sequence_audit/C5_tested_peak_sequence_covariates.csv.gz"
STAGE={"220403_C2":"E12.5","220403_D2":"E12.5","220403_A2":"E13.5","220403_B2":"E13.5","220327_A1":"E15.5","220327_A2":"E15.5"}
ORDER=list(STAGE)
a=ad.read_h5ad(H5); assert a.shape==(17787,248047) and sparse.issparse(a.X)
X=a.X.tocsr(); assert np.isfinite(X.data).all() and (X.data>=0).all()
obs=pd.read_csv(OBS).sort_values("canonical_index").reset_index(drop=True); lab=pd.read_csv(LAB).sort_values("canonical_index").reset_index(drop=True)
assert (obs.obs_name==lab.obs_name).all(); obs["frozen_cluster"]=lab.cluster.to_numpy(); obs["stage"]=obs.slice_id.map(STAGE)
rows=pd.Index(a.obs_names.astype(str)).get_indexer(obs.obs_name.astype(str)); assert (rows>=0).all()
X=X[rows]; lib=np.asarray(X.sum(1)).ravel(); detected=X.getnnz(1)
cov=pd.read_csv(COV); assert len(cov)==6540 and cov.peak_id.is_unique
testidx=pd.Index(a.var_names.astype(str)).get_indexer(cov.peak_id); assert (testidx>=0).all()
Xt=X[:,testidx].astype(np.float32); scale=np.divide(1e4,lib,out=np.zeros_like(lib,float),where=lib>0)
N=Xt.multiply(scale[:,None]).tocsr(); N.data=np.log1p(N.data)
mods={}
for m,f,n in [("EarlyHigh","C5_Module_EarlyHigh_peaks.csv",126),("LateHigh","C5_Module_LateHigh_peaks.csv",106)]:
 z=pd.read_csv(MOD/f); assert len(z)==n and z.peak_id.is_unique and set(z.peak_id)<=set(cov.peak_id); mods[m]=z.peak_id.tolist()
union=set(mods["EarlyHigh"])|set(mods["LateHigh"]); pool=np.flatnonzero(~cov.peak_id.isin(union).to_numpy())
Z=cov[["mean_accessibility","GC_fraction","peak_length"]].to_numpy(float); Z=(Z-Z.mean(0))/Z.std(0)
rng=np.random.default_rng(1234); tree=cKDTree(Z[pool]+rng.normal(0,1e-12,(len(pool),3)))
scores={}; matches=[]
for m,peaks in mods.items():
 mi=pd.Index(cov.peak_id).get_indexer(peaks); dist,nn=tree.query(Z[mi],k=5); bgidx=pool[np.asarray(nn).ravel()]
 for q,(p,dv,js) in enumerate(zip(peaks,np.atleast_2d(dist),np.atleast_2d(pool[np.asarray(nn)]))):
  for rank,(d,j) in enumerate(zip(dv,js),1): matches.append({"module":m,"module_peak":p,"module_peak_index":int(mi[q]),"background_peak":cov.iloc[int(j)].peak_id,"background_peak_index":int(j),"match_rank":rank,"covariate_distance":float(d)})
 unadj=np.asarray(N[:,mi].mean(1)).ravel(); bg=np.asarray(N[:,bgidx].mean(1)).ravel()
 scores[m]={"unadjusted":unadj,"background":bg,"adjusted":unadj-bg,"matched_indices":bgidx}
pd.DataFrame(matches).to_csv(BM/"module_peak_matched_background_peaks.csv",index=False)
pd.DataFrame([{"module":m,"module_peak_count":len(mods[m]),"controls_per_peak":5,"match_count":len(scores[m]["matched_indices"]),
 "module_peaks_with_complete_matches":len(mods[m]),"match_success_fraction":1.0,"unique_background_peak_count":len(set(scores[m]["matched_indices"]))} for m in mods]).to_csv(BM/"background_match_summary.csv",index=False)
slice_depth=pd.Series(lib).groupby(obs.slice_id).transform("sum").to_numpy()
out=pd.DataFrame({"canonical_index":obs.canonical_index.to_numpy(),"slice":obs.slice_id.to_numpy(),"stage":obs.stage.to_numpy(),"frozen_cluster":obs.frozen_cluster.to_numpy(),
 "spatial_x":obs.spatial_x.to_numpy(),"spatial_y":obs.spatial_y.to_numpy(),"library_size":lib,"detected_peak_count":detected,"slice_total_depth":slice_depth,
 "EarlyHigh_module_score":scores["EarlyHigh"]["adjusted"],"LateHigh_module_score":scores["LateHigh"]["adjusted"],
 "Late_minus_Early_score":scores["LateHigh"]["adjusted"]-scores["EarlyHigh"]["adjusted"],
 "EarlyHigh_unadjusted_score":scores["EarlyHigh"]["unadjusted"],"LateHigh_unadjusted_score":scores["LateHigh"]["unadjusted"],
 "EarlyHigh_background_score":scores["EarlyHigh"]["background"],"LateHigh_background_score":scores["LateHigh"]["background"],
 "is_C5":obs.frozen_cluster.eq(5).to_numpy()},index=obs.obs_name)
bad={c:int((~np.isfinite(out[c].to_numpy(float))).sum()) for c in out.select_dtypes("number")}
bad={k:v for k,v in bad.items() if v}
assert len(out)==17787 and not bad,bad
out.to_csv(SCO/"C5_spot_module_scores.csv.gz",index_label="obs_name",compression="gzip")
Xscore=out[["EarlyHigh_module_score","LateHigh_module_score","Late_minus_Early_score"]].to_numpy(np.float32)
sad=ad.AnnData(Xscore,obs=out.copy(),var=pd.DataFrame(index=["EarlyHigh_module_score","LateHigh_module_score","Late_minus_Early_score"]))
sad.obsm["spatial"]=out[["spatial_x","spatial_y"]].to_numpy(); sad.write_h5ad(SCO/"C5_spot_module_scores.h5ad",compression="gzip")
trace={"status":"COMPLETE","spot_count":len(out),"C5_spot_count":int(out.is_C5.sum()),"tested_peak_count":len(cov),
 "module_peak_counts":{m:len(v) for m,v in mods.items()},"library_normalization":"scale 10000 then natural log1p",
 "background_matching":"5 nearest tested non-module peaks per module peak by standardized mean accessibility, GC, length; replacement; seed 1234",
 "score":"module normalized mean - multiplicity-preserving matched-background normalized mean"}
(META/"module_score_build_trace.json").write_text(json.dumps(trace,indent=2)+"\n")
print(json.dumps(trace,indent=2))

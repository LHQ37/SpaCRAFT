#!/usr/bin/env python3
"""Build raw C5/C6 cluster-by-slice pseudobulks from frozen integer peak counts."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import hashlib, json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/accessibility_phase2_v1"
PB=OUT/"pseudobulk"; QC=OUT/"qc"; META=OUT/"metadata"
for d in (PB,QC,META,OUT/"C5",OUT/"C6",OUT/"combined"): d.mkdir(parents=True,exist_ok=True)
H5=DATA_ROOT / "multi_slice_unzipped/多切片/Mouse_Embryo/mouse_embryo.h5ad"
PEAKS=DATA_ROOT / "Mouse_Embryo_6slice_experiments/run_20260723_112344/prepared_data/filtered_peak_names.txt"
OBS=MULTISLICE_RESULTS_ROOT / "canonical/six_slices_obs.csv"
LAB=MULTISLICE_RESULTS_ROOT / "six_slices/SpaCRAFT/cluster_labels.csv"
ELIG=ROOT/"results/composition_v1/six_slice/cluster_downstream_eligibility.csv"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGE={"220403_C2":"E12.5","220403_D2":"E12.5","220403_A2":"E13.5",
       "220403_B2":"E13.5","220327_A1":"E15.5","220327_A2":"E15.5"}

elig=pd.read_csv(ELIG).set_index("cluster")
assert bool(elig.loc[5,"stage_associated_peak_analysis_eligible"])
assert bool(elig.loc[6,"stage_associated_peak_analysis_eligible"])
obs=pd.read_csv(OBS).sort_values("canonical_index")
lab=pd.read_csv(LAB).sort_values("canonical_index")
assert len(obs)==len(lab)==17787
assert not obs.canonical_index.duplicated().any() and not lab.canonical_index.duplicated().any()
assert (obs.obs_name.to_numpy()==lab.obs_name.to_numpy()).all()
obs["cluster"]=lab.cluster.to_numpy()
obs["stage"]=obs.slice_id.map(STAGE)
assert obs.stage.notna().all() and sorted(obs.cluster.unique())==list(range(11))

a=ad.read_h5ad(H5)
assert a.n_obs==17787 and sparse.issparse(a.X)
assert np.isfinite(a.X.data).all() and (a.X.data>=0).all() and np.allclose(a.X.data,np.rint(a.X.data))
assert not pd.Index(a.obs_names).duplicated().any() and not pd.Index(a.var_names).duplicated().any()
raw_row=pd.Index(a.obs_names.astype(str)).get_indexer(obs.obs_name.astype(str))
assert (raw_row>=0).all() and len(np.unique(raw_row))==17787
peaks=pd.Index(PEAKS.read_text().splitlines())
assert len(peaks)==8282 and not peaks.duplicated().any()
peak_col=pd.Index(a.var_names.astype(str)).get_indexer(peaks)
assert (peak_col>=0).all() and len(np.unique(peak_col))==8282
X=a.X.tocsr()[raw_row][:,peak_col]
assert X.shape==(17787,8282) and np.isfinite(X.data).all()

sample_rows=[]
trace={"h5ad":str(H5),"formal_peak_file":str(PEAKS),
       "formal_peak_file_sha256":hashlib.sha256(PEAKS.read_bytes()).hexdigest(),
       "canonical_obs":str(OBS),"frozen_labels":str(LAB),"shape":[17787,8282],
       "clusters":[5,6],"slice_order":ORDER}
for cluster in (5,6):
    raw=np.zeros((8282,6),dtype=np.int64)
    detection=np.zeros((8282,6),dtype=float)
    for j,s in enumerate(ORDER):
        mask=(obs.cluster.eq(cluster)&obs.slice_id.eq(s)).to_numpy()
        xs=X[mask]
        sums=np.asarray(xs.sum(axis=0)).ravel()
        assert np.allclose(sums,np.rint(sums))
        raw[:,j]=sums.astype(np.int64)
        detection[:,j]=np.asarray((xs>0).sum(axis=0)).ravel()/xs.shape[0]
        per_spot=np.asarray(xs.sum(axis=1)).ravel()
        lib=int(sums.sum())
        sample_rows.append({"cluster":cluster,"slice":s,"stage":STAGE[s],
            "spot_count":int(xs.shape[0]),"total_peak_counts":lib,
            "detected_peak_count":int((sums>0).sum()),
            "median_counts_per_spot":float(np.median(per_spot)),
            "zero_fraction":float((sums==0).mean()),"library_size":lib})
    rdf=pd.DataFrame(raw,columns=ORDER); rdf.insert(0,"peak_id",peaks)
    ddf=pd.DataFrame(detection,columns=ORDER); ddf.insert(0,"peak_id",peaks)
    rdf.to_csv(PB/f"C{cluster}_raw_pseudobulk_counts.csv.gz",index=False,compression="gzip")
    ddf.to_csv(QC/f"C{cluster}_spot_detection_fraction.csv.gz",index=False,compression="gzip")
pd.DataFrame(sample_rows).to_csv(PB/"pseudobulk_sample_metadata.csv",index=False)
(META/"FROZEN_INPUT_TRACE.json").write_text(json.dumps(trace,indent=2)+"\n")
print(json.dumps({"status":"PASS","shape":[17787,8282],"clusters":[5,6]},indent=2))

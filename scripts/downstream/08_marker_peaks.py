#!/usr/bin/env python3
"""One-vs-rest marker peaks using SnapATAC2's formal likelihood-ratio test."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import norm
from snapatac2.tools._diff import _diff_test_helper, _p_adjust_bh

ROOT=SINGLE_DOWNSTREAM_ROOT
objects=sorted((ROOT/"data_cache").glob("*/standardized.h5ad"))
if len(sys.argv)>1:
    requested=set(sys.argv[1:])
    objects=[x for x in objects if x.parent.name in requested]
    if not objects:
        raise SystemExit("No requested dataset found")
for obj in objects:
    name=obj.parent.name
    out=ROOT/"results/by_slice"/name/"marker_peaks"
    out.mkdir(parents=True,exist_ok=True)
    a=ad.read_h5ad(obj,backed="r")
    x=a.layers["counts"] if "counts" in a.layers else a.X
    if hasattr(x, "to_memory"):
        x=x.to_memory()
    x=x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)
    groups=a.obs["SpaCRAFT_cluster"].astype(str).values
    all_tables=[]
    for cluster in sorted(pd.unique(groups),key=lambda z:(len(z),z)):
        inside=np.flatnonzero(groups==cluster); outside=np.flatnonzero(groups!=cluster)
        m1=x[inside].tocsc(); m2=x[outside].tocsc()
        f1=np.asarray((m1>0).sum(axis=0)).ravel()/len(inside)
        f2=np.asarray((m2>0).sum(axis=0)).ravel()/len(outside)
        c1=np.asarray(m1.sum(axis=0)).ravel()+1
        c2=np.asarray(m2.sum(axis=0)).ravel()+1
        rpm1=c1/(c1.sum()/1e6); rpm2=c2/(c2.sum()/1e6)
        logfc=np.log2(rpm1/rpm2)
        selected=np.flatnonzero((np.maximum(f1,f2)>=0.05)&(np.abs(logfc)>=0.5))
        p=np.asarray(_diff_test_helper(sparse.vstack([m1,m2]).tocsr(),
                                      np.r_[np.zeros(len(inside)),np.ones(len(outside))],
                                      selected,None))
        q=_p_adjust_bh(p)
        z=norm.isf(np.maximum(p/2,np.finfo(float).tiny))*np.sign(logfc[selected])
        tab=pd.DataFrame({"cluster":cluster,"peak_id":a.var_names[selected],
            "log2FC":logfc[selected],"test_statistic_signed_z":z,"pvalue":p,
            "adjusted_pvalue":q,"detection_fraction_in":f1[selected],
            "detection_fraction_out":f2[selected]})
        tab["direction"]=np.where(tab.log2FC>0,"positive","negative")
        tab=tab.sort_values(["adjusted_pvalue","log2FC"],ascending=[True,False])
        tab["rank"]=np.arange(1,len(tab)+1)
        all_tables.append(tab)
    result=pd.concat(all_tables,ignore_index=True)
    result.to_csv(out/"all_marker_peak_statistics.csv.gz",index=False,compression="gzip")
    result[result.direction.eq("positive")].to_csv(out/"positive_markers.csv.gz",index=False,compression="gzip")
    result[result.direction.eq("negative")].to_csv(out/"negative_markers.csv.gz",index=False,compression="gzip")
    summary=(result.query("direction == 'positive' and adjusted_pvalue <= 0.05")
             .groupby("cluster").size().rename("significant_positive_marker_count").reset_index())
    summary.to_csv(out/"marker_counts_by_cluster.csv",index=False)
    print(f"{name}: rows={len(result)}")
print("marker_peak_status=PASS")

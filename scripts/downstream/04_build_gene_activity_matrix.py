#!/usr/bin/env python3
"""Build exact-rule sparse multi-slice gene activity on primary and sensitivity axes."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
from collections import defaultdict
import hashlib, json, re, resource
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/gene_activity_phase4_v1"; MAT=OUT/"matrices"; PB=OUT/"pseudobulk"
SENS=OUT/"sensitivity"; META=OUT/"metadata"; CHUNK=MAT/"chunks"
for d in (MAT,PB,SENS,META,CHUNK,OUT/"C0",OUT/"C5",OUT/"C6",
          OUT/"module_concordance",OUT/"spatial_candidates"): d.mkdir(parents=True,exist_ok=True)
H5=DATA_ROOT / "multi_slice_unzipped/多切片/Mouse_Embryo/mouse_embryo.h5ad"
GTF=GENOME_ROOT / "annotation/gencode.vM25.annotation.gtf"
OBS=MULTISLICE_RESULTS_ROOT / "canonical/six_slices_obs.csv"
LAB=MULTISLICE_RESULTS_ROOT / "six_slices/SpaCRAFT/cluster_labels.csv"
SEL=DATA_ROOT / "Mouse_Embryo_6slice_experiments/run_20260723_112344/prepared_data/filtered_peak_names.txt"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGE={"220403_C2":"E12.5","220403_D2":"E12.5","220403_A2":"E13.5",
       "220403_B2":"E13.5","220327_A1":"E15.5","220327_A2":"E15.5"}
CHUNK_SIZE=2000

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def attrs(s): return {m.group(1):m.group(2) for m in re.finditer(r'(\S+) "([^"]*)";',s)}
genes=[]
allowed={f"chr{i}" for i in range(1,20)}|{"chrX","chrY"}
with GTF.open() as f:
    for line in f:
        if line.startswith("#"): continue
        z=line.rstrip().split("\t")
        if len(z)<9 or z[0] not in allowed or z[2]!="gene": continue
        a=attrs(z[8]); gid=a.get("gene_id","").split(".")[0]
        if not gid: continue
        start=int(z[3])-1; end=int(z[4]); strand=z[6]
        genes.append({"gene_id":gid,"gene_symbol":a.get("gene_name",gid),"chromosome":z[0],
                      "start":start,"end":end,"strand":strand,
                      "tss":start if strand=="+" else end-1})
g=pd.DataFrame(genes).drop_duplicates("gene_id").reset_index(drop=True)
assert g.gene_id.is_unique and g.gene_symbol.ne("").all()
bychr={}
for chrom,x in g.groupby("chromosome"):
    ii=x.index.to_numpy(); oo=np.argsort(g.loc[ii,"tss"].to_numpy())
    bychr[chrom]=(g.loc[ii[oo],"tss"].to_numpy(),ii[oo])

def parse_peak(s):
    m=re.match(r"^([^:]+)[:-](\d+)-(\d+)$",str(s))
    if not m: return None
    return m.group(1),int(m.group(2)),int(m.group(3))

a=ad.read_h5ad(H5)
assert a.shape==(17787,248047) and sparse.issparse(a.X)
assert np.isfinite(a.X.data).all() and (a.X.data>=0).all() and np.allclose(a.X.data,np.rint(a.X.data))
assert pd.Index(a.var_names).is_unique
obs=pd.read_csv(OBS).sort_values("canonical_index").reset_index(drop=True)
lab=pd.read_csv(LAB).sort_values("canonical_index").reset_index(drop=True)
assert len(obs)==len(lab)==17787 and (obs.obs_name==lab.obs_name).all()
obs["frozen_cluster"]=lab.cluster.to_numpy(); obs["stage"]=obs.slice_id.map(STAGE)
srcrow=pd.Index(a.obs_names.astype(str)).get_indexer(obs.obs_name.astype(str))
assert (srcrow>=0).all() and len(np.unique(srcrow))==17787

peak_rows=[]; pi=[]; gj=[]
for i,pid in enumerate(a.var_names.astype(str)):
    p=parse_peak(pid)
    if p is None or p[0] not in bychr: continue
    chrom,s,e=p; mid=(s+e)//2; tss,ii=bychr[chrom]; k=np.searchsorted(tss,mid)
    cand=np.unique(np.clip(np.arange(k-4,k+5),0,len(ii)-1))
    j=int(ii[cand[np.argmin(np.abs(tss[cand]-mid))]])
    signed=(mid-int(g.loc[j,"tss"]))*(1 if g.loc[j,"strand"]=="+" else -1)
    within=(-2000<=signed<=0) or (s<int(g.loc[j,"end"]) and e>int(g.loc[j,"start"]))
    if within:
        pi.append(i); gj.append(j)
        peak_rows.append({"peak_id":pid,"peak_index":i,"gene_index":j,
          "gene_id":g.loc[j,"gene_id"],"gene_symbol":g.loc[j,"gene_symbol"],
          "distance_to_tss":signed,"mapping_region":"gene_body_or_2kb_upstream"})
peakmap=pd.DataFrame(peak_rows)
keepgenes=np.sort(peakmap.gene_index.unique())
old2new={old:new for new,old in enumerate(keepgenes)}
peakmap["gene_index_compact"]=peakmap.gene_index.map(old2new)
gv=g.loc[keepgenes].reset_index(drop=True).copy()
gv["peaks_contributing"]=peakmap.groupby("gene_index_compact").size().reindex(range(len(gv))).to_numpy()
gv["scoring_rule"]="nearest_TSS_gene; gene_body_or_strand_aware_2kb_upstream; one_peak_once; no_weight"
peakmap.to_csv(META/"primary_peak_to_gene_mapping.csv.gz",index=False,compression="gzip")
mapping=sparse.csr_matrix((np.ones(len(pi),dtype=np.float32),
    (np.asarray(pi),peakmap.gene_index_compact.to_numpy())),shape=(a.n_vars,len(gv)))
selected=pd.Index(SEL.read_text().splitlines())
selidx=pd.Index(a.var_names.astype(str)).get_indexer(selected)
assert len(selected)==8282 and (selidx>=0).all()
sel_lookup={old:i for i,old in enumerate(selidx)}
sp=peakmap[peakmap.peak_index.isin(sel_lookup)].copy()
sens_n=sp.groupby("gene_index_compact").size().reindex(range(len(gv)),fill_value=0).to_numpy()
sensmap=sparse.csr_matrix((np.ones(len(sp),dtype=np.float32),
    ([sel_lookup[x] for x in sp.peak_index],sp.gene_index_compact)),
    shape=(8282,len(gv)))

X=a.X.tocsr()
raw_chunks=[]; norm_chunks=[]
sens_sum=np.zeros((len(gv),6),float); sens_det=np.zeros((len(gv),6),float)
for start in range(0,len(obs),CHUNK_SIZE):
    end=min(start+CHUNK_SIZE,len(obs)); rows=srcrow[start:end]
    xc=X[rows]
    raw=(xc@mapping).tocsr().astype(np.float32)
    lib=np.asarray(raw.sum(axis=1)).ravel()
    scale=np.divide(1e4,lib,out=np.zeros_like(lib,dtype=float),where=lib>0)
    norm=raw.multiply(scale[:,None]).tocsr(); norm.data=np.log1p(norm.data)
    sparse.save_npz(CHUNK/f"raw_{start:05d}_{end:05d}.npz",raw,compressed=True)
    sparse.save_npz(CHUNK/f"normalized_{start:05d}_{end:05d}.npz",norm,compressed=True)
    raw_chunks.append(raw); norm_chunks.append(norm)
    sr=(xc[:,selidx]@sensmap).tocsr()
    o=obs.iloc[start:end]
    for j,s in enumerate(ORDER):
        mask=o.slice_id.eq(s).to_numpy() & o.frozen_cluster.eq(5).to_numpy()
        if mask.any():
            sens_sum[:,j]+=np.asarray(sr[mask].sum(axis=0)).ravel()
            sens_det[:,j]+=np.asarray((sr[mask]>0).sum(axis=0)).ravel()
raw=sparse.vstack(raw_chunks,format="csr")
norm=sparse.vstack(norm_chunks,format="csr")
assert raw.shape==norm.shape==(17787,len(gv))
assert np.isfinite(raw.data).all() and np.isfinite(norm.data).all() and (raw.data>=0).all()
adobs=pd.DataFrame({"canonical_index":obs.canonical_index.to_numpy(),
  "section":obs.slice_id.to_numpy(),"stage":obs.stage.to_numpy(),
  "frozen_cluster":obs.frozen_cluster.to_numpy()},index=obs.obs_name)
advar=gv.set_index("gene_id")[["gene_symbol","chromosome","start","end","strand",
                               "peaks_contributing","scoring_rule"]]
advar["gene_id"]=advar.index.astype(str)
rawad=ad.AnnData(raw,obs=adobs.copy(),var=advar.copy())
normad=ad.AnnData(norm,obs=adobs.copy(),var=advar.copy())
coords=obs[["spatial_x","spatial_y"]].to_numpy()
rawad.obsm["spatial"]=coords; normad.obsm["spatial"]=coords
rawad.write_h5ad(MAT/"six_slice_gene_activity_raw.h5ad",compression="gzip")
normad.write_h5ad(MAT/"six_slice_gene_activity_normalized.h5ad",compression="gzip")

sample=[]; primary_tables={}
for cluster in (0,5,6):
    sums=np.zeros((len(gv),6),float); det=np.zeros((len(gv),6),float)
    for j,s in enumerate(ORDER):
        mask=obs.slice_id.eq(s).to_numpy()&obs.frozen_cluster.eq(cluster).to_numpy()
        rr=raw[mask]
        sums[:,j]=np.asarray(rr.sum(axis=0)).ravel()
        det[:,j]=np.asarray((rr>0).sum(axis=0)).ravel()/mask.sum()
        sample.append({"cluster":cluster,"slice":s,"stage":STAGE[s],"spot_count":int(mask.sum()),
          "total_gene_activity":float(sums[:,j].sum()),"detected_gene_count":int((sums[:,j]>0).sum()),
          "normalization_factor":float(1e6/sums[:,j].sum())})
    libs=sums.sum(axis=0); log=np.log2(sums/libs*1e6+1)
    tab=gv[["gene_id","gene_symbol","peaks_contributing"]].copy()
    for j,s in enumerate(ORDER):
        tab[f"{s}_raw_sum"]=sums[:,j]; tab[f"{s}_log2CPM"]=log[:,j]
        tab[f"{s}_spot_detection_fraction"]=det[:,j]
    tab.to_csv(PB/f"C{cluster}_gene_activity_by_slice.csv.gz",index=False,compression="gzip")
    primary_tables[cluster]=(sums,log,det)
pd.DataFrame(sample).to_csv(PB/"sample_metadata.csv",index=False)

sens_lib=sens_sum.sum(axis=0)
sens_log=np.log2(sens_sum/sens_lib*1e6+1)
stab=gv[["gene_id","gene_symbol"]].copy()
stab["peaks_contributing"]=sens_n
c5counts=obs[obs.frozen_cluster.eq(5)].groupby("slice_id").size()
for j,s in enumerate(ORDER):
    stab[f"{s}_raw_sum"]=sens_sum[:,j]; stab[f"{s}_log2CPM"]=sens_log[:,j]
    stab[f"{s}_spot_detection_fraction"]=sens_det[:,j]/c5counts[s]
stab.to_csv(SENS/"C5_8282_gene_activity_by_slice.csv.gz",index=False,compression="gzip")

trace={"primary_h5ad":str(H5),"primary_peak_count":248047,"sensitivity_peak_file":str(SEL),
 "sensitivity_peak_count":8282,"gtf":str(GTF),"gtf_sha256":sha256_file(GTF),
 "canonical_spots":17787,"gene_count":len(gv),"mapped_peak_count":len(peakmap),
 "chunk_size_spots":CHUNK_SIZE,"chunk_count":len(raw_chunks),
 "peak_memory_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 "method_reuse_status":"BIOLOGICALLY_EQUIVALENT_IMPLEMENTATION"}
(META/"GENE_ACTIVITY_BUILD_TRACE.json").write_text(json.dumps(trace,indent=2)+"\n")
print(json.dumps(trace,indent=2))

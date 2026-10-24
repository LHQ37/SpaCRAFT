#!/usr/bin/env python3
"""Resume the frozen six-slice analysis after marker peaks.

This script never writes formal inputs, standardized objects, or marker results.
All analyses use fixed parameters declared below and in downstream_config.yaml.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, math, os, platform, re, subprocess, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = SINGLE_DOWNSTREAM_ROOT
GTF = GENOME_ROOT / "annotation/gencode.vM25.annotation.gtf"
FASTA = GENOME_ROOT / "genome.fa"
MEME = GENOME_ROOT / "motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
SEED = 1234
RNG = np.random.default_rng(SEED)
PROM_UP, PROM_DOWN = 2000, 500
GA_UP, GA_DOWN = 2000, 0
ANNOT_CATS = ["promoter", "5' UTR", "exon", "intron", "downstream", "distal/intergenic"]
SLICES = sorted(p.parent.name for p in (ROOT/"data_cache").glob("*/standardized.h5ad"))

def sha256(p: Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while b:=f.read(chunk): h.update(b)
    return h.hexdigest()

def atomic_json(p: Path, x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+"\n")
    q.replace(p)

def parse_peak(s):
    m=re.match(r"^([^:]+)[:-](\d+)-(\d+)$",str(s))
    if not m: return None
    return m.group(1),int(m.group(2)),int(m.group(3))

def parse_attrs(s):
    return {m.group(1):m.group(2) for m in re.finditer(r'(\S+) "([^"]*)";',s)}

def load_genes():
    """Read GENCODE once; gene-level model and merged feature intervals."""
    genes={}
    with open(GTF) as f:
        for line in f:
            if line.startswith("#"): continue
            z=line.rstrip().split("\t")
            if len(z)<9 or z[0] not in {f"chr{i}" for i in range(1,20)}|{"chrX","chrY"}: continue
            typ=z[2]
            if typ not in {"gene","exon","five_prime_UTR"}: continue
            a=parse_attrs(z[8]); gid=a.get("gene_id","").split(".")[0]
            if not gid: continue
            if gid not in genes:
                genes[gid]={"gene_id":gid,"gene_name":a.get("gene_name",gid),"chrom":z[0],
                            "start":int(z[3])-1,"end":int(z[4]),"strand":z[6],"exons":[],"utr5":[]}
            g=genes[gid]
            if typ=="gene":
                g.update(start=int(z[3])-1,end=int(z[4]),gene_name=a.get("gene_name",gid),strand=z[6],chrom=z[0])
            elif typ=="exon": g["exons"].append((int(z[3])-1,int(z[4])))
            else: g["utr5"].append((int(z[3])-1,int(z[4])))
    genes=[g for g in genes.values() if g["end"]>g["start"]]
    bychr=defaultdict(list)
    for i,g in enumerate(genes):
        g["tss"]=g["start"] if g["strand"]=="+" else g["end"]-1
        bychr[g["chrom"]].append(i)
    idx={}
    for chrom,ii in bychr.items():
        ii=np.array(ii,dtype=int); order=np.argsort([genes[i]["tss"] for i in ii])
        ii=ii[order]; idx[chrom]=(np.array([genes[i]["tss"] for i in ii]),ii)
    return genes,idx

def interval_overlap(s,e,intervals):
    return any(s < b and e > a for a,b in intervals)

def annotate_peaks(peaks,genes,gidx):
    rows=[]
    for peak in peaks:
        p=parse_peak(peak)
        if p is None:
            rows.append(dict(peak_id=peak,chromosome="",start=np.nan,end=np.nan,
                gene_id="",gene_name="",distance_to_tss=np.nan,annotation="unannotated",reason="parse_failure"))
            continue
        chrom,s,e=p; mid=(s+e)//2
        if chrom not in gidx:
            rows.append(dict(peak_id=peak,chromosome=chrom,start=s,end=e,gene_id="",gene_name="",
                distance_to_tss=np.nan,annotation="unannotated",reason="no_reference_gene_on_chromosome"))
            continue
        tss,ii=gidx[chrom]; k=np.searchsorted(tss,mid)
        cand=np.unique(np.clip(np.arange(k-4,k+5),0,len(ii)-1))
        gi=ii[cand[np.argmin(np.abs(tss[cand]-mid))]]; g=genes[int(gi)]
        signed=(mid-g["tss"])*(1 if g["strand"]=="+" else -1)
        prom=(-PROM_UP <= signed <= PROM_DOWN)
        if prom: cat="promoter"
        elif interval_overlap(s,e,g["utr5"]): cat="5' UTR"
        elif interval_overlap(s,e,g["exons"]): cat="exon"
        elif s<g["end"] and e>g["start"]: cat="intron"
        elif 0 < signed <= 2000: cat="downstream"
        else: cat="distal/intergenic"
        rows.append(dict(peak_id=peak,chromosome=chrom,start=s,end=e,gene_id=g["gene_id"],
            gene_name=g["gene_name"],gene_strand=g["strand"],gene_start=g["start"],gene_end=g["end"],
            distance_to_tss=int(signed),annotation=cat,reason=""))
    return pd.DataFrame(rows)

def validate_resume():
    rows=[]; ok=True
    for name in SLICES:
        obj=ROOT/"data_cache"/name/"standardized.h5ad"; a=ad.read_h5ad(obj,backed="r")
        clusters=set(a.obs["SpaCRAFT_cluster"].astype(str)); peaks=set(a.var_names)
        rec={"slice":name,"object":str(obj),"object_sha256":sha256(obj),"n_spots":a.n_obs,"n_peaks":a.n_vars}
        for key,fn in [("all","all_marker_peak_statistics.csv.gz"),("positive","positive_markers.csv.gz"),("negative","negative_markers.csv.gz")]:
            p=ROOT/"results/by_slice"/name/"marker_peaks"/fn
            if not p.exists(): ok=False; rec[key+"_status"]="MISSING"; continue
            x=pd.read_csv(p)
            required={"cluster","peak_id","adjusted_pvalue","pvalue","log2FC","direction"}
            valid=required.issubset(x.columns) and set(x.cluster.astype(str))==clusters and set(x.peak_id).issubset(peaks)
            nums=x.select_dtypes(include=[np.number])
            valid=valid and np.isfinite(nums.to_numpy()).all()
            if key=="positive": valid=valid and x.direction.eq("positive").all()
            if key=="negative": valid=valid and x.direction.eq("negative").all()
            rec.update({key+"_status":"PASS" if valid else "FAIL",key+"_rows":len(x),key+"_sha256":sha256(p)})
            ok &= valid
        rows.append(rec)
    status="MARKER_PEAK_OUTPUTS_VALID" if ok and len(SLICES)==6 else "MARKER_PEAK_OUTPUTS_INVALID"
    pd.DataFrame(rows).to_csv(ROOT/"audit/RESUME_MARKER_PEAK_DETAILS.csv",index=False)
    text=["# Resume from marker peaks","",f"- Status: **{status}**",
          f"- Slice count: {len(SLICES)}","- Marker peaks recomputed: **False**",
          "- Checks: positive/negative/full tables; BH field; frozen cluster set; exact peak-ID subset; finite numeric values; SHA256 provenance.",
          "","Detailed per-slice checks: `RESUME_MARKER_PEAK_DETAILS.csv`."]
    (ROOT/"audit/RESUME_FROM_MARKER_PEAKS.md").write_text("\n".join(text)+"\n")
    if status!="MARKER_PEAK_OUTPUTS_VALID": raise SystemExit(status)

def savefig(fig,p):
    p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,bbox_inches="tight",dpi=180); plt.close(fig)

def plot_matrix(df,index,col,val,title,p,cmap="viridis"):
    if df.empty: return
    mat=df.pivot_table(index=index,columns=col,values=val,aggfunc="mean",fill_value=0)
    fig,ax=plt.subplots(figsize=(max(5,.55*len(mat.columns)+2),max(3,.35*len(mat)+2)))
    im=ax.imshow(mat.values,aspect="auto",cmap=cmap); ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=60,ha="right",fontsize=7)
    ax.set_yticks(range(len(mat.index)),mat.index,fontsize=7); ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.7)
    savefig(fig,p)

def moran_i(values,coords,k=8):
    v=np.asarray(values,float); xy=np.asarray(coords,float)
    if len(v)<4 or np.nanstd(v)==0:return np.nan
    kk=min(k+1,len(v)); _,nn=cKDTree(xy).query(xy,k=kk); nn=np.atleast_2d(nn)[:,1:]
    z=v-v.mean(); den=(z*z).sum()
    return float(len(v)/nn.size * sum(z[i]*z[j] for i in range(len(v)) for j in nn[i])/den)

def annotation_one(name,a,ann):
    out=ROOT/"results/by_slice"/name/"peak_annotation"; out.mkdir(parents=True,exist_ok=True)
    ann.to_csv(out/"all_peaks_annotation.csv",index=False)
    stats=pd.read_csv(ROOT/"results/by_slice"/name/"marker_peaks/all_marker_peak_statistics.csv.gz")
    ma=stats.merge(ann,on="peak_id",how="left"); ma.to_csv(out/"marker_peaks_annotation.csv",index=False)
    sig=ma.query("direction=='positive' and adjusted_pvalue<=0.05 and log2FC>=0.5")
    summ=(sig.groupby(["cluster","annotation"]).size().rename("count").reset_index())
    summ["proportion"]=summ.groupby("cluster")["count"].transform(lambda x:x/x.sum())
    summ.to_csv(out/"cluster_annotation_summary.csv",index=False)
    bg=ann.annotation.value_counts().rename("background_count").to_frame()
    bg["background_proportion"]=bg.background_count/bg.background_count.sum()
    cmp=(sig.groupby(["cluster","annotation"]).size().rename("marker_count").reset_index().merge(bg,left_on="annotation",right_index=True,how="left"))
    cmp["marker_proportion"]=cmp.groupby("cluster").marker_count.transform(lambda x:x/x.sum())
    cmp["fold_vs_background"]=(cmp.marker_proportion+1e-9)/(cmp.background_proportion+1e-9)
    cmp.to_csv(out/"annotation_background_comparison.csv",index=False)
    atomic_json(out/"metadata.json",{"status":"COMPLETE","build":"mm10/GRCm38","promoter_definition":"TSS upstream 2000 bp to downstream 500 bp",
      "annotation_priority":ANNOT_CATS,"all_peaks":len(ann),"unannotated":int((ann.annotation=="unannotated").sum()),
      "unannotated_reasons":ann.loc[ann.annotation=="unannotated","reason"].value_counts().to_dict(),"marker_definition":"all tested marker rows retained"})
    fig,ax=plt.subplots(figsize=(7,4)); pv=summ.pivot(index="cluster",columns="annotation",values="proportion").fillna(0)
    pv.reindex(columns=ANNOT_CATS,fill_value=0).plot.bar(stacked=True,ax=ax); ax.set_ylabel("proportion"); ax.set_title(f"{name}: marker peak annotation")
    savefig(fig,out/"cluster_annotation_stacked_bar.png")
    plot_matrix(summ,"cluster","annotation","proportion",f"{name}: annotation proportions",out/"cluster_annotation_proportion_heatmap.png")
    cmp.to_csv(out/"cluster_annotation_stacked_bar_plot_data.csv",index=False)

def gene_activity_one(name,a,ann,genes):
    out=ROOT/"results/by_slice"/name/"gene_activity"; out.mkdir(parents=True,exist_ok=True)
    # A peak is assigned once to its nearest gene if it overlaps gene body or the fixed strand-aware upstream domain.
    dist=ann.distance_to_tss.to_numpy(float); within=ann.gene_id.ne("").to_numpy() & (
        ((dist>=-GA_UP)&(dist<=GA_DOWN)) |
        ((ann.start.to_numpy(float)<ann.gene_end.to_numpy(float))&(ann.end.to_numpy(float)>ann.gene_start.to_numpy(float))))
    gtab=ann.loc[within,["gene_id","gene_name"]].drop_duplicates("gene_id").reset_index(drop=True)
    gid_to_j={g:i for i,g in enumerate(gtab.gene_id)}
    pi=np.flatnonzero(within); gj=np.array([gid_to_j[x] for x in ann.loc[within,"gene_id"]])
    mapping=sparse.csr_matrix((np.ones(len(pi)),(pi,gj)),shape=(a.n_vars,len(gtab)))
    X=a.layers["counts"] if "counts" in a.layers else a.X
    if hasattr(X,"to_memory"): X=X.to_memory()
    X=sparse.csr_matrix(X,dtype=np.float32)
    raw=X@mapping
    lib=np.asarray(raw.sum(1)).ravel(); scale=np.divide(1e4,lib,out=np.zeros_like(lib,float),where=lib>0)
    ga=raw.multiply(scale[:,None]).tocsr(); ga.data=np.log1p(ga.data)
    ga_ad=ad.AnnData(ga,obs=a.obs[["SpaCRAFT_cluster","ground_truth_annotation"]].copy(),var=gtab.set_index("gene_id"))
    ga_ad.obsm["spatial"]=np.asarray(a.obsm["spatial"]); ga_ad.write_h5ad(out/"gene_activity.h5ad",compression="gzip")
    pd.DataFrame({"gene_id":gtab.gene_id,"gene_name":gtab.gene_name}).to_csv(out/"genes.csv",index=False)
    pd.DataFrame({"spot_id":a.obs_names}).to_csv(out/"spots.csv",index=False)
    clusters=a.obs["SpaCRAFT_cluster"].astype(str).to_numpy(); means=[]
    marker=[]; spatial=[]
    for c in sorted(set(clusters),key=lambda z:(len(z),z)):
        inside=clusters==c; mi=np.asarray(ga[inside].mean(0)).ravel(); mo=np.asarray(ga[~inside].mean(0)).ravel()
        deti=np.asarray((raw[inside]>0).mean(0)).ravel(); deto=np.asarray((raw[~inside]>0).mean(0)).ravel()
        for j,row in gtab.iterrows():
            means.append((c,row.gene_id,row.gene_name,mi[j]))
        cand=np.argsort(mi-mo)[-min(300,len(mi)):]; ps=[]
        for j in cand:
            u=mannwhitneyu(ga[inside,j].toarray().ravel(),ga[~inside,j].toarray().ravel(),alternative="greater").pvalue
            ps.append(u)
        qs=multipletests(ps,method="fdr_bh")[1] if ps else []
        for j,pv,qv in zip(cand,ps,qs):
            vals=ga[:,j].toarray().ravel(); mor= moran_i(vals,a.obsm["spatial"])
            marker.append((c,gtab.iloc[j].gene_id,gtab.iloc[j].gene_name,mi[j],mo[j],mi[j]-mo[j],pv,qv,deti[j],deto[j],mor))
            spatial.append((c,gtab.iloc[j].gene_id,gtab.iloc[j].gene_name,mor))
    mdf=pd.DataFrame(marker,columns=["cluster","gene_id","gene_name","mean_in","mean_out","effect_size","pvalue","adjusted_pvalue","detection_fraction_in","detection_fraction_out","morans_i"])
    pd.DataFrame(means,columns=["cluster","gene_id","gene_name","mean_activity"]).to_csv(out/"cluster_gene_activity.csv",index=False)
    mdf.to_csv(out/"gene_activity_marker_statistics.csv",index=False)
    pd.DataFrame(spatial,columns=["cluster","gene_id","gene_name","morans_i"]).to_csv(out/"gene_activity_spatial_statistics.csv",index=False)
    mdf["candidate_score"]=mdf.effect_size.rank(pct=True)+(-np.log10(mdf.adjusted_pvalue.clip(1e-300))).rank(pct=True)+mdf.detection_fraction_in.rank(pct=True)+mdf.morans_i.fillna(-1).rank(pct=True)
    top=(mdf.query("effect_size>0 and detection_fraction_in>=0.05").sort_values(["candidate_score","adjusted_pvalue"],ascending=[False,True])
         .drop_duplicates("gene_id").head(20)); top.to_csv(out/"top20_candidate_genes.csv",index=False)
    atomic_json(out/"metadata.json",{"status":"COMPLETE_MATRIX_BASED","implementation":"fixed peak-to-nearest-gene sparse aggregation; SnapATAC2 make_gene_matrix audited but fragment-dependent",
      "include_gene_body":True,"promoter_upstream_bp":GA_UP,"promoter_downstream_bp":GA_DOWN,"distal_max_distance_bp":0,
      "distance_weighting":"none","cross_gene_overlapping_peak_policy":"nearest gene only","normalization":"library size 10000","log_transform":"log1p","smoothing":False,
      "ground_truth_used":False})
    heat=mdf.sort_values("candidate_score",ascending=False).drop_duplicates(["cluster","gene_id"]).groupby("cluster").head(5)
    plot_matrix(heat,"gene_name","cluster","mean_in",f"{name}: gene activity",out/"cluster_gene_activity_heatmap.png","magma")
    heat.to_csv(out/"cluster_gene_activity_heatmap_plot_data.csv",index=False)
    fig,axs=plt.subplots(1,4,figsize=(14,3)); coords=np.asarray(a.obsm["spatial"])
    for ax,(_,r) in zip(axs,top.head(4).iterrows()):
        j=gid_to_j[r.gene_id]; val=ga[:,j].toarray().ravel(); sc=ax.scatter(coords[:,0],coords[:,1],c=val,s=3,cmap="viridis"); ax.set_title(r.gene_name); ax.set_aspect("equal"); ax.axis("off")
    savefig(fig,out/"top4_gene_activity_spatial_maps.png"); top.head(4).to_csv(out/"top4_gene_activity_spatial_maps_plot_data.csv",index=False)
    return ga,gtab,mdf

def parse_meme(limit=None):
    from snapatac2._snapatac2 import PyDNAMotif
    lines=MEME.read_text().splitlines(); motifs=[]; names={}
    i=0
    while i<len(lines):
        if lines[i].startswith("MOTIF "):
            z=lines[i].split(maxsplit=2); mid=z[1]; names[mid]=z[2] if len(z)>2 else mid
            i+=1
            while i<len(lines) and "letter-probability matrix:" not in lines[i]: i+=1
            if i>=len(lines): break
            w=int(re.search(r"\bw=\s*(\d+)",lines[i]).group(1)); mat=[]
            for q in lines[i+1:i+1+w]: mat.append([float(x) for x in q.split()[:4]])
            motifs.append(PyDNAMotif(mid,np.asarray(mat)))
            i+=w
        i+=1
    return motifs[:limit] if limit else motifs,names

def motif_one(name,ann):
    out=ROOT/"results/by_slice"/name/"motif_enrichment"; out.mkdir(parents=True,exist_ok=True)
    stats=pd.read_csv(ROOT/"results/by_slice"/name/"marker_peaks/positive_markers.csv.gz")
    sig=stats.query("adjusted_pvalue<=0.05 and log2FC>=0.5")
    fasta_chroms=set(pd.read_csv(str(FASTA)+".fai",sep="\t",header=None,usecols=[0])[0].astype(str))
    valid_peaks=set(ann.loc[ann.chromosome.isin(fasta_chroms),"peak_id"])
    regions={}
    for c,x in sig.groupby(stats.cluster.astype(str)):
        x=x[x.peak_id.isin(valid_peaks)].sort_values(["adjusted_pvalue","log2FC"],ascending=[True,False]).head(400)
        regions[str(c)]=[re.sub(r"^([^:]+)-(\d+)-(\d+)$",r"\1:\2-\3",p) for p in x.peak_id]
    # Background is detectable peaks from the same slice, sampled reproducibly and stratified approximately by length.
    bgdf=ann.query("annotation!='unannotated' and chromosome in @fasta_chroms").copy(); bgdf["length"]=bgdf.end-bgdf.start
    bg=[]
    for _,x in bgdf.groupby(pd.qcut(bgdf["length"],min(10,bgdf["length"].nunique()),duplicates="drop")):
        take=min(len(x),max(1,5000//10)); bg.extend(x.sample(take,random_state=SEED).peak_id)
    bg=[re.sub(r"^([^:]+)-(\d+)-(\d+)$",r"\1:\2-\3",p) for p in bg]
    try:
        import snapatac2 as snap
        motifs,names=parse_meme()
        res=snap.tl.motif_enrichment(motifs,regions,FASTA,background=bg,method="binomial")
        tables=[]
        for c,x in res.items():
            d=x.to_pandas().rename(columns={"id":"motif_id","name":"motif_name","family":"motif_family","log2(fold change)":"log2_fold_change","p-value":"pvalue","adjusted p-value":"adjusted_pvalue"})
            d["cluster"]=c; d["motif_name"]=d.motif_id.map(names).fillna(d.motif_name); tables.append(d)
        full=pd.concat(tables,ignore_index=True)
        full.to_csv(out/"full_motif_enrichment.csv",index=False)
        full["family_display"]=full.motif_name.str.replace(r"\(.*","",regex=True).str.split("::").str[0].str.upper()
        nr=full.sort_values(["adjusted_pvalue","log2_fold_change"],ascending=[True,False]).drop_duplicates(["cluster","family_display"])
        nr.to_csv(out/"nonredundant_motif_enrichment.csv",index=False)
        top=nr.query("log2_fold_change>0").groupby("cluster").head(10); top.to_csv(out/"cluster_top_motifs.csv",index=False)
        atomic_json(out/"motif_background_metadata.json",{"status":"COMPLETE","motif_database":str(MEME),"motif_count":len(motifs),"foreground":"significant positive marker peaks; max 400 per cluster",
          "background":"same-slice detectable peaks, fixed seed length-stratified sample","background_n":len(bg),"GC_matching":"scanner method does not provide exact GC matching; length stratification used","seed":SEED,"multiple_testing":"BH by SnapATAC2"})
        plot_matrix(top,"cluster","motif_name","log2_fold_change",f"{name}: motif enrichment",out/"cluster_motif_enrichment_heatmap.png","coolwarm")
        top.to_csv(out/"cluster_motif_enrichment_heatmap_plot_data.csv",index=False)
        return full
    except Exception as e:
        for fn in ["full_motif_enrichment.csv","nonredundant_motif_enrichment.csv","cluster_top_motifs.csv"]:
            pd.DataFrame(columns=["cluster","motif_id","motif_name","log2_fold_change","pvalue","adjusted_pvalue"]).to_csv(out/fn,index=False)
        atomic_json(out/"motif_background_metadata.json",{"status":"FAILED","error":repr(e),"no_proxy_substitution":True})
        return pd.DataFrame()

def common_figures(name,a):
    fdir=ROOT/"figures/per_slice"/name; fdir.mkdir(parents=True,exist_ok=True)
    xy=np.asarray(a.obsm["spatial"]); fig,axs=plt.subplots(1,2,figsize=(8,4))
    for ax,key,title in [(axs[0],"ground_truth_annotation","Ground truth"),(axs[1],"SpaCRAFT_cluster","SpaCRAFT clusters")]:
        lab=a.obs[key].astype(str); codes=pd.Categorical(lab).codes
        ax.scatter(xy[:,0],xy[:,1],c=codes,s=4,cmap="tab20"); ax.set_title(title); ax.axis("off"); ax.set_aspect("equal")
    savefig(fig,fdir/"ground_truth_vs_spacraft_clusters.png")
    pd.DataFrame({"spot_id":a.obs_names,"x":xy[:,0],"y":xy[:,1],"ground_truth":a.obs.ground_truth_annotation.astype(str),"SpaCRAFT_cluster":a.obs.SpaCRAFT_cluster.astype(str)}).to_csv(fdir/"ground_truth_vs_spacraft_clusters_plot_data.csv",index=False)
    st=pd.read_csv(ROOT/"results/by_slice"/name/"marker_peaks/positive_markers.csv.gz").query("adjusted_pvalue<=0.05")
    top=st.sort_values(["cluster","adjusted_pvalue"]).groupby(st.cluster.astype(str)).head(10)
    plot_matrix(top,"peak_id","cluster","log2FC",f"{name}: marker peaks",fdir/"marker_peak_heatmap.png","magma")
    top.to_csv(fdir/"marker_peak_heatmap_plot_data.csv",index=False)

def unavailable_outputs(name):
    # chromVAR is never substituted; no validated fragment-dependent output is created.
    out=ROOT/"results/by_slice"/name/"motif_activity"; out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"metadata.json",{"status":"ENVIRONMENT_UNAVAILABLE","chromVAR_run":False,"proxy_used":False})
    co=ROOT/"results/by_slice"/name/"coaccessibility"; co.mkdir(parents=True,exist_ok=True)
    atomic_json(co/"metadata.json",{"status":"SKIPPED_INSUFFICIENT_SUPPORT","reason":"fixed candidate-locus stability analysis not supported without completing candidate validation"})
    ps=ROOT/"results/by_slice"/name/"pseudotime"; ps.mkdir(parents=True,exist_ok=True)
    atomic_json(ps/"metadata.json",{"status":"NOT_SUPPORTED","reason":"motif activity unavailable and no pre-specified biological root; spatial direction was not used"})
    fp=ROOT/"results/by_slice"/name/"footprinting"; fp.mkdir(parents=True,exist_ok=True)
    atomic_json(fp/"metadata.json",{"footprinting_status":"SKIPPED_NO_VALIDATED_FRAGMENTS"})

def placeholder_go(name,ann,mdf):
    out=ROOT/"results/by_slice"/name/"functional_enrichment"; out.mkdir(parents=True,exist_ok=True)
    mapping=ann[["peak_id","gene_id","gene_name"]].drop_duplicates(); mapping.to_csv(out/"gene_mapping_table.csv",index=False)
    bg=sorted(set(mapping.gene_name)-{""}); (out/"background_gene_set.txt").write_text("\n".join(bg)+"\n")
    cols=["cluster","term_id","term_name","overlap_count","pvalue","adjusted_pvalue","genes"]
    for fn in ["full_GO_BP_results.csv","filtered_GO_BP_results.csv","cluster_GO_summary.csv"]:
        pd.DataFrame(columns=cols).to_csv(out/fn,index=False)
    atomic_json(out/"metadata.json",{"status":"ENVIRONMENT_UNAVAILABLE","reason":"No validated local mouse GO association resource; no online or whole-genome-background substitute used",
      "intended_background":"all detectable mapped peak genes","method_name":"GO enrichment (not GREAT)","filter":"BH adjusted p-value <0.05; max 5 nonredundant terms/cluster"})

def make_previews():
    prev=ROOT/"figures/previews"; prev.mkdir(parents=True,exist_ok=True)
    specs={
      "all_slices_marker_peak_preview.png":("figures/per_slice","marker_peak_heatmap.png"),
      "all_slices_peak_annotation_preview.png":("results/by_slice","peak_annotation/cluster_annotation_stacked_bar.png"),
      "all_slices_gene_activity_preview.png":("results/by_slice","gene_activity/cluster_gene_activity_heatmap.png"),
      "all_slices_motif_enrichment_preview.png":("results/by_slice","motif_enrichment/cluster_motif_enrichment_heatmap.png")}
    for fn,(base,rel) in specs.items():
        paths=[ROOT/base/n/rel for n in SLICES if (ROOT/base/n/rel).exists()]
        if not paths: continue
        fig,axs=plt.subplots(2,3,figsize=(15,9)); axs=axs.ravel()
        for ax,n,p in zip(axs,SLICES,paths): ax.imshow(plt.imread(p)); ax.set_title(n); ax.axis("off")
        for ax in axs[len(paths):]: ax.axis("off")
        savefig(fig,prev/fn)

def score_and_reports(metrics):
    out=ROOT/"results/screening"; out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(metrics)
    comps=["Marker_Peak_Strength","Marker_Peak_Significance","Heatmap_Block_Structure","Peak_Annotation_Completeness",
      "Gene_Activity_Specificity","Gene_Activity_Spatial_MoransI","Motif_Enrichment_Strength","Motif_Activity_Spatial_MoransI",
      "Motif_Enrichment_Activity_Concordance","Functional_Enrichment_Specificity","Coaccessibility_Support","Pseudotime_Support","Biological_Interpretability"]
    df["Analysis_Coverage"]=df[comps].notna().mean(axis=1)
    for c in comps:
        x=df[c]; df[c+"_scaled"]=np.nan if x.notna().sum()==0 else (x-x.min())/(x.max()-x.min()+1e-12)*100
    scaled=[c+"_scaled" for c in comps]
    df["Available_Component_Mean"]=df[scaled].mean(axis=1,skipna=True)
    df.to_csv(out/"DOWNSTREAM_SCREENING_SCORECARD.csv",index=False)
    rank=df.sort_values(["Available_Component_Mean","Analysis_Coverage"],ascending=False).copy(); rank.insert(0,"rank",range(1,len(rank)+1))
    rank[["rank","slice","Available_Component_Mean","Analysis_Coverage"]].to_csv(out/"SLICE_RANKING.csv",index=False)
    df[["slice"]+comps].notna().assign(slice=df.slice).to_csv(out/"ANALYSIS_COMPLETENESS_MATRIX.csv",index=False)
    rank[["rank","slice","Available_Component_Mean","Analysis_Coverage"]].to_csv(out/"CANDIDATE_PANEL_RANKING.csv",index=False)
    main,backup=rank.iloc[0]["slice"],rank.iloc[1]["slice"]
    (ROOT/"audit/RECOMMENDED_MAIN_SLICE.md").write_text(f"# Recommended main slice\n\n- main_slice: **{main}**\n- backup_slice: **{backup}**\n- basis: pre-frozen available-component mean with Analysis_Coverage reported separately.\n")
    panels="a Ground truth vs SpaCRAFT clusters; b Marker peak heatmap; c Gene activity spatial maps; d Motif enrichment heatmap; e Peak genomic annotation; f Gene activity heatmap"
    (ROOT/"audit/RECOMMENDED_MAIN_FIGURE_PLAN.md").write_text(f"# Recommended main figure plan\n\n- main_slice: **{main}**\n- recommended_panel_set: {panels}\n- final_figure_assembled: **False**\n")
    (ROOT/"audit/BACKUP_PANEL_PLAN.md").write_text(f"# Backup panel plan\n\n- backup_slice: **{backup}**\n- backup_panel_set: Ground truth/clusters; marker peaks; gene activity maps; motif enrichment; peak annotation; gene activity heatmap.\n")
    (ROOT/"audit/FAILED_OR_WEAK_ANALYSES.md").write_text("# Failed or weak analyses\n\n- chromVAR/motif activity: ENVIRONMENT_UNAVAILABLE; no proxy used.\n- GO BP: ENVIRONMENT_UNAVAILABLE because no validated local mouse GO association resource was present.\n- matrix-based co-accessibility: SKIPPED_INSUFFICIENT_SUPPORT.\n- pseudotime: NOT_SUPPORTED.\n- footprinting: SKIPPED_NO_VALIDATED_FRAGMENTS.\n")
    return main,backup

def manifest():
    files=[p for p in ROOT.rglob("*") if p.is_file() and "marker_peaks" not in p.parts and p.name!="resume_after_marker_peaks.log"]
    rows=[{"path":str(p.relative_to(ROOT)),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(files)]
    pd.DataFrame(rows).to_csv(ROOT/"audit/RESUME_OUTPUT_MANIFEST.csv",index=False)
    (ROOT/"audit/RESUME_OUTPUT_SHA256.txt").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows))

def main():
    validate_resume()
    genes,gidx=load_genes()
    metrics=[]
    for name in SLICES:
        print(f"[{datetime.now().isoformat()}] slice={name}",flush=True)
        obj=ROOT/"data_cache"/name/"standardized.h5ad"; a=ad.read_h5ad(obj)
        ann=annotate_peaks(a.var_names,genes,gidx)
        annotation_one(name,a,ann)
        ga,gtab,mdf=gene_activity_one(name,a,ann,genes)
        motif=motif_one(name,ann)
        placeholder_go(name,ann,mdf); unavailable_outputs(name); common_figures(name,a)
        st=pd.read_csv(ROOT/"results/by_slice"/name/"marker_peaks/positive_markers.csv.gz")
        sig=st.query("adjusted_pvalue<=0.05 and log2FC>=0.5")
        metrics.append({"slice":name,
          "Marker_Peak_Strength":float(sig.log2FC.median()) if len(sig) else np.nan,
          "Marker_Peak_Significance":float((-np.log10(sig.adjusted_pvalue.clip(1e-300))).median()) if len(sig) else np.nan,
          "Heatmap_Block_Structure":float(sig.groupby("cluster").log2FC.mean().mean()) if len(sig) else np.nan,
          "Peak_Annotation_Completeness":float((ann.annotation!="unannotated").mean()),
          "Gene_Activity_Specificity":float(mdf.effect_size.nlargest(min(50,len(mdf))).mean()),
          "Gene_Activity_Spatial_MoransI":float(mdf.morans_i.nlargest(min(50,len(mdf))).mean()),
          "Motif_Enrichment_Strength":float(motif.query("adjusted_pvalue<=0.05 and log2_fold_change>0").log2_fold_change.mean()) if len(motif) else np.nan,
          "Motif_Activity_Spatial_MoransI":np.nan,"Motif_Enrichment_Activity_Concordance":np.nan,
          "Functional_Enrichment_Specificity":np.nan,"Coaccessibility_Support":np.nan,"Pseudotime_Support":np.nan,
          "Biological_Interpretability":float(mdf.query("adjusted_pvalue<=0.05 and effect_size>0").gene_id.nunique())})
    make_previews(); main_slice,backup=score_and_reports(metrics)
    atomic_json(ROOT/"audit/RESUME_FINAL_STATUS.json",{"resume_status":"COMPLETE_WITH_DOCUMENTED_UNAVAILABLE_COMPONENTS","main_slice":main_slice,"backup_slice":backup,
      "models_retrained":False,"embedding_recomputed":False,"clusters_recomputed":False,"formal_inputs_modified":False,"final_figure_assembled":False})
    manifest()
    print(f"recommended_main_slice={main_slice}\nrecommended_backup_slice={backup}",flush=True)

if __name__=="__main__": main()

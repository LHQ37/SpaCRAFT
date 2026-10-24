#!/usr/bin/env python3
"""Render the formal SpaCRAFT single-slice downstream figure.

This script performs no fitting and does not change any source result. It reads
frozen cluster labels/counts and applies deterministic plotting transforms.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import re
import textwrap

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PNG = OUT_DIR / "Figure_single_slice_downstream.png"
PDF = OUT_DIR / "Figure_single_slice_downstream.pdf"
ATAC = REPO / "data/E15_5-S1_atac.h5ad"
RESULT = REPO / "results/single_slice/spacraft/results/E15_5-S1_atac"
SINGLE = REPO / "results/single_slice/benchmark/oracle_cluster_labels"
MULTI_ROOT = Path(os.environ.get(
    "SPACRAFT_MULTISLICE_COMPARISON_ROOT",
    REPO / "results/multi_slice",
))
GENOME_ROOT = Path(os.environ.get(
    "SPACRAFT_GENOME_ROOT",
    REPO / "resources/mm10",
))
MULTI = MULTI_ROOT / "e11_e18"
GTF = GENOME_ROOT / "annotation/gencode.vM25.annotation.gtf"
MEME = GENOME_ROOT / "motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"

METHODS = ["SpaCRAFT", "INSTINCT", "GraphST", "PeakVI", "scDREAMER", "SEDR", "STAligner"]
CLUSTERS = list(range(7))
CLUSTER_COLORS = dict(zip(CLUSTERS, mpl.colormaps["tab10"](np.arange(7))))
SELECTED_MOTIFS = ["MA0510.3", "MA0703.3", "MA0507.3", "MA0768.3"]
MOTIF_NAMES = {"MA0510.3":"RFX5", "MA0703.3":"LMX1B", "MA0507.3":"POU2F2", "MA0768.3":"Lef1"}
SELECTED_GENES = ["Nfix", "Foxg1", "Zfhx2os", "Sox1"]
HEATMAP_GENES = ["Phyhipl", "Bicdl1", "Rarg", "Ccnd3", "Gm10800", "CT010467.1",
                 "Gm4657", "Neurod6", "Gm36101", "Gm7480", "4930404A05Rik",
                 "Myh15", "Actc1", "Gm47523"]
MARKER_PEAKS = {
    0:["chr11:74589885-74590386","chr5:37241725-37242226","chr6:91153464-91153965","chr9:108824994-108825495","chr9:122572152-122572653"],
    1:["chr2:128698525-128699026","chr8:104022944-104023445","chr17:46073865-46074366","chr9:21496650-21497151","chr6:4505304-4505805"],
    2:["chr12:80002373-80002874","chr9:90692885-90693386","chr9:91355594-91356095","chr14:61330446-61330947","chr4:8799466-8799967"],
    3:["chr12:6638752-6639253","chr3:89539955-89540456","chr17:87745217-87745718","chr10:82662456-82662957","chr12:12220630-12221131"],
    4:["chr9:7796112-7796613","chr1:157937736-157938237","chr13:91296706-91297207","chr8:11896105-11896606","chr18:65563820-65564321"],
    5:["chr7:73890389-73890890","chr5:131596822-131597323","chr18:39214928-39215429","chr18:36356336-36356837","chr16:49303542-49304043"],
    6:["chr7:29123690-29124191","chr7:142550854-142551355","chr7:142600861-142601362","chr10:69208164-69208665","chr10:80914109-80914610"],
}

# Complete seven-cluster values used by the formal figure.
MOTIF_VALUES = {
 "RFX5":[4.000259864541359,-0.4249706677053072,1.2160621685376036,0.4451935641955292,0.2064410001890041,0.5520149533963853,-4.35729987021505],
 "LMX1B":[1.9101370666291728,0.3215678264272833,5.009311660203008,4.2538970289370255,4.602285274988539,3.13977013072188,0.3215678264272833],
 "POU2F2":[0.081284726782524,-3.3962714397906915,1.6129962530898594,1.879524420666954,3.756739241880542,-0.5780691354960951,-1.0671208264850909],
 "Lef1":[1.5688635907228423,-0.0417312220280521,1.8278103074530765,0.8622831342136951,1.4207839222386065,4.117197688725774,0.8622831342136951],
}

ENRICHMENT = [
 ("BP",1,"animal gross anatomical part developmental process",41.4245298585689,"Tgfb1, Bmp4, Notch1, Wnt5a"),
 ("BP",0,"synaptic signaling",28.70695398413356,"Drd2, Reln, Nlgn3, Grin1"),
 ("BP",6,"muscle structure development",18.976682336495987,"Notch1, Nog, Vegfa, Igf1"),
 ("CC",0,"synapse",63.00031171403619,"Gria2, Grin1, Gria1, Grin2b"),
 ("CC",1,"extracellular matrix",32.5343355695842,"Itgb1, Anxa1, Anxa2, Anxa5"),
 ("CC",6,"myofibril",30.237565778469826,"Vcl, Dag1, Palld, Pdlim4"),
 ("MF",0,"gated channel activity",14.767030887291266,"Gria2, Grin1, Gria1, Grik2"),
 ("MF",1,"extracellular matrix structural constituent",13.605201893889037,"Thbs1, Slit2, Agrn, Comp"),
 ("MF",1,"actin binding",12.97474295932438,"Egfr, Lrrk2, Prkce, Ptk2"),
 ("KEGG",6,"Cytoskeleton in muscle cells",24.2696808360144,"Itgb1, Actb, Itgb3, Itga5"),
 ("KEGG",1,"Integrin signaling",11.17928138667431,"Pik3ca, Pik3cb, Pik3cd, Pik3r2"),
 ("KEGG",0,"Axon guidance",7.297891745191247,"Gnai2, Camk2a, Camk2b, Camk2g"),
]
CATEGORY_COLORS = {"BP":"#7698BC", "CC":"#F7A149", "MF":"#7AB674", "KEGG":"#EB7A79"}

mpl.rcParams.update({"font.family":"DejaVu Sans", "font.size":7.2, "axes.titlesize":9.2,
 "axes.labelsize":7.6, "xtick.labelsize":6.6, "ytick.labelsize":6.6,
 "legend.fontsize":6.2, "axes.linewidth":.65, "pdf.fonttype":42,
 "ps.fonttype":42, "figure.facecolor":"white"})


def require_inputs():
    paths = [ATAC, GTF, MEME, RESULT/"spatial_coordinates.csv", RESULT/"cluster_labels.csv",
             RESULT/"ground_truth_labels.csv"]
    paths += [SINGLE/m/"E15_5-S1_atac.csv" for m in ("INSTINCT","PeakVI")]
    paths += [MULTI/m/"cluster_labels.csv" for m in ("GraphST","scDREAMER","SEDR","STAligner")]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError("Missing formal input(s):\n"+"\n".join(missing))


def load_spatial():
    xy=pd.read_csv(RESULT/"spatial_coordinates.csv").rename(columns={"obs_name":"spot_id","spatial_1":"x","spatial_2":"y"})
    cl=pd.read_csv(RESULT/"cluster_labels.csv").rename(columns={"obs_name":"spot_id","cluster_label":"SpaCRAFT"})
    gt=pd.read_csv(RESULT/"ground_truth_labels.csv").rename(columns={"obs_name":"spot_id"})
    return xy.merge(gt,on="spot_id",validate="one_to_one").merge(cl,on="spot_id",validate="one_to_one")


def load_method(method, spatial):
    if method=="SpaCRAFT": return spatial[["spot_id","SpaCRAFT"]].rename(columns={"SpaCRAFT":"cluster"})
    if method in ("INSTINCT","PeakVI"):
        q=pd.read_csv(SINGLE/method/"E15_5-S1_atac.csv")
        return q.rename(columns={"obs_name":"spot_id","cluster_label":"cluster"})[["spot_id","cluster"]]
    q=pd.read_csv(MULTI/method/"cluster_labels.csv")
    q=q[q.obs_name.str.startswith("E15_5-S1#")].copy()
    q["spot_id"]=q.obs_name.str.extract(r"^E15_5-S1#(.+?)_E15_S1$")[0]
    return q[["spot_id","cluster"]]


def matched_methods(spatial):
    gt_order=list(dict.fromkeys(spatial.ground_truth.astype(str)))
    gt_colors=dict(zip(gt_order,mpl.colormaps["tab20"](np.linspace(0,.82,len(gt_order)))))
    out={}
    for method in METHODS:
        q=spatial[["spot_id","x","y","ground_truth"]].merge(load_method(method,spatial),on="spot_id",validate="one_to_one")
        if len(q)!=len(spatial): raise ValueError(f"{method}: incomplete spot join")
        tab=pd.crosstab(q.cluster,q.ground_truth).reindex(columns=gt_order,fill_value=0)
        rr,cc=linear_sum_assignment(-tab.to_numpy())
        mapping={tab.index[r]:tab.columns[c] for r,c in zip(rr,cc)}
        q["matched_label"]=q.cluster.map(mapping).fillna("Unmatched")
        q["color"]=q.matched_label.map(gt_colors).fillna("#9a9a9a")
        out[method]=q
    return gt_order,gt_colors,out


def parse_attrs(s):
    return {m.group(1):m.group(2) for m in re.finditer(r'(\S+) "([^"]*)";',s)}


def gene_activity(a):
    genes={}
    with GTF.open() as fh:
        for line in fh:
            if line.startswith("#"): continue
            z=line.rstrip().split("\t")
            if len(z)<9 or z[2]!="gene": continue
            aa=parse_attrs(z[8]); gid=aa.get("gene_id","").split(".")[0]
            if gid: genes[gid]={"gene_id":gid,"gene_name":aa.get("gene_name",gid),"chrom":z[0],
                                "start":int(z[3])-1,"end":int(z[4]),"strand":z[6]}
    genes=list(genes.values()); bychr=defaultdict(list)
    for i,g in enumerate(genes):
        g["tss"]=g["start"] if g["strand"]=="+" else g["end"]-1; bychr[g["chrom"]].append(i)
    index={}
    for chrom,ii in bychr.items():
        ii=np.asarray(ii); order=np.argsort([genes[i]["tss"] for i in ii]); ii=ii[order]
        index[chrom]=(np.asarray([genes[i]["tss"] for i in ii]),ii)
    rows=[]
    for peak in a.var_names:
        m=re.match(r"^([^:]+)[:-](\d+)-(\d+)$",str(peak))
        if not m: rows.append(("","",False)); continue
        chrom,s,e=m.group(1),int(m.group(2)),int(m.group(3)); mid=(s+e)//2
        if chrom not in index: rows.append(("","",False)); continue
        tss,ii=index[chrom]; k=np.searchsorted(tss,mid)
        cand=np.unique(np.clip(np.arange(k-4,k+5),0,len(ii)-1)); gi=ii[cand[np.argmin(np.abs(tss[cand]-mid))]]
        g=genes[int(gi)]; signed=(mid-g["tss"])*(1 if g["strand"]=="+" else -1)
        within=(-2000<=signed<=0) or (s<g["end"] and e>g["start"])
        rows.append((g["gene_id"],g["gene_name"],within))
    ann=pd.DataFrame(rows,columns=["gene_id","gene_name","within"])
    gtab=ann.loc[ann.within,["gene_id","gene_name"]].drop_duplicates("gene_id").reset_index(drop=True)
    gid_to_j={g:i for i,g in enumerate(gtab.gene_id)}; pi=np.flatnonzero(ann.within)
    gj=np.asarray([gid_to_j[x] for x in ann.loc[ann.within,"gene_id"]])
    mapping=sparse.csr_matrix((np.ones(len(pi)),(pi,gj)),shape=(a.n_vars,len(gtab)))
    raw=sparse.csr_matrix(a.X,dtype=np.float32)@mapping
    lib=np.asarray(raw.sum(1)).ravel(); scale=np.divide(1e4,lib,out=np.zeros_like(lib,float),where=lib>0)
    ga=raw.multiply(scale[:,None]).tocsr(); ga.data=np.log1p(ga.data)
    return ga,gtab


def marker_matrix(a, clusters):
    X=sparse.csr_matrix(a.X); var={x:i for i,x in enumerate(a.var_names)}; rows=[]
    for c in CLUSTERS:
        inside=clusters==c; outside=~inside
        m1=X[inside].tocsc(); m2=X[outside].tocsc()
        c1=np.asarray(m1.sum(0)).ravel()+1; c2=np.asarray(m2.sum(0)).ravel()+1
        rpm1=c1/(c1.sum()/1e6); rpm2=c2/(c2.sum()/1e6); logfc=np.log2(rpm1/rpm2)
        rows.extend(logfc[var[p]] for p in MARKER_PEAKS[c])
    mat=np.zeros((35,7));
    for i,c in enumerate(np.repeat(CLUSTERS,5)): mat[i,c]=rows[i]
    return mat


def parse_meme():
    wanted=set(SELECTED_MOTIFS); lines=MEME.read_text().splitlines(); out={}; i=0
    while i<len(lines):
        if lines[i].startswith("MOTIF "):
            parts=lines[i].split(); mid=parts[1]; j=i+1
            while j<len(lines) and "letter-probability matrix" not in lines[j]: j+=1
            if mid in wanted and j<len(lines):
                w=int(lines[j].split("w=")[1].split()[0]); out[mid]=np.asarray([[float(x) for x in z.split()[:4]] for z in lines[j+1:j+1+w]])
            i=j
        i+=1
    if set(out)!=wanted: raise ValueError("Selected motif PWM missing")
    return out


BASE_COL={"A":"#2ca02c","C":"#1f77b4","G":"#f2a900","T":"#d62728"}
def draw_logo(ax,pwm):
    eps=1e-9; ic=2+np.sum(pwm*np.log2(pwm+eps),axis=1); heights=pwm*ic[:,None]
    fp=FontProperties(family="DejaVu Sans",weight="bold")
    for x,row in enumerate(heights):
        y=0
        for idx in np.argsort(row):
            base="ACGT"[idx]; h=row[idx]
            if h<=0: continue
            tp=TextPath((0,0),base,size=1,prop=fp); bb=tp.get_extents()
            trans=Affine2D().scale(.86/max(bb.width,eps),h/max(bb.height,eps)).translate(x+.07,y)
            ax.add_patch(mpl.patches.PathPatch(tp,transform=trans+ax.transData,color=BASE_COL[base],lw=0)); y+=h
    ax.set_xlim(0,len(pwm)); ax.set_ylim(0,2.05); ax.set_xticks([]); ax.set_yticks([0,2]); ax.tick_params(length=1,pad=1)
    ax.spines[["top","right"]].set_visible(False)


def scatter_map(ax,q,colors,title,s=2.5):
    ax.scatter(q.x,q.y,c=colors,s=s,lw=0,rasterized=True); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title,pad=1.2)


def panel_a(fig,spec,spatial,gt_order,gt_colors,methods):
    sub=spec.subgridspec(4,4,height_ratios=[1,1,.12,.34],wspace=.06,hspace=.055); axes=[]
    gt=spatial.copy(); gt_color=gt.ground_truth.map(gt_colors)
    order=[("Ground truth",gt,gt_color)]+[(m,methods[m],methods[m].color) for m in METHODS]
    for i,(name,q,col) in enumerate(order):
        ax=fig.add_subplot(sub[i//4,i%4]); scatter_map(ax,q,col,name); axes.append(ax)
    cluster_leg=fig.add_subplot(sub[2,:]); cluster_leg.axis("off")
    cluster_handles=[Line2D([0],[0],marker="o",ls="",ms=4,color=CLUSTER_COLORS[c],label=f"C{c}") for c in CLUSTERS]
    cluster_leg.legend(handles=cluster_handles,loc="center",ncol=7,frameon=False,
                       title="SpaCRAFT clusters",columnspacing=.75,handletextpad=.2,title_fontsize=6.6)
    gt_leg=fig.add_subplot(sub[3,:]); gt_leg.axis("off")
    gt_handles=[Line2D([0],[0],marker="o",ls="",ms=3.5,color=gt_colors[x],label=x.replace("_"," ")) for x in gt_order]
    gt_handles.append(Line2D([0],[0],marker="o",ls="",ms=3.5,color="#9a9a9a",label="Unmatched cluster"))
    gt_leg.legend(handles=gt_handles,loc="center",ncol=4,frameon=False,
                  title="Ground-truth labels / matched-cluster colors",columnspacing=.65,handletextpad=.18,title_fontsize=6.6)
    return axes+[cluster_leg,gt_leg]


def panel_b(fig,spec,mat):
    sub=spec.subgridspec(1,3,width_ratios=[.33,1,.045],wspace=.03); lab=fig.add_subplot(sub[0]); ax=fig.add_subplot(sub[1]); cax=fig.add_subplot(sub[2])
    im=ax.imshow(mat,aspect="auto",cmap="magma",interpolation="nearest"); ax.set_xticks(CLUSTERS,[f"C{x}" for x in CLUSTERS]); ax.set_yticks([])
    for y in np.arange(5,35,5)-.5: ax.axhline(y,color="white",lw=.5)
    lab.set_xlim(0,1); lab.set_ylim(34.5,-.5); lab.axis("off")
    for i in range(35): lab.text(.98,i,f"C{i//5}_peak{i%5+1}",ha="right",va="center",fontsize=5.25)
    ax.set_title("Marker peaks",pad=4); fig.colorbar(im,cax=cax).set_label("Marker log2FC",labelpad=2)
    return [lab,ax,cax]


def panel_c(fig,spec,spatial,pwms):
    outer=spec.subgridspec(1,2,width_ratios=[1,.025],wspace=.035); paired=outer[0].subgridspec(2,2,wspace=.10,hspace=.12)
    vals=np.asarray(list(MOTIF_VALUES.values())); vmin,vmax=float(vals.min()),float(vals.max()); axes=[]; sc=None
    for i,mid in enumerate(SELECTED_MOTIFS):
        unit=paired[i//2,i%2].subgridspec(2,1,height_ratios=[.16,1],hspace=.008)
        logo=fig.add_subplot(unit[0]); draw_logo(logo,pwms[mid]); logo.set_title(MOTIF_NAMES[mid],pad=1,fontweight="bold",fontsize=8.2); axes.append(logo)
        ax=fig.add_subplot(unit[1]); name=MOTIF_NAMES[mid]; c=spatial.SpaCRAFT.map(dict(enumerate(MOTIF_VALUES[name])))
        sc=ax.scatter(spatial.x,spatial.y,c=c,s=3.3,cmap="viridis",vmin=vmin,vmax=vmax,lw=0,rasterized=True)
        ax.set_aspect("equal"); ax.axis("off"); axes.append(ax)
    slot=outer[1].subgridspec(3,1,height_ratios=[.19,.62,.19]); cax=fig.add_subplot(slot[1]); cb=fig.colorbar(sc,cax=cax)
    cb.set_label("Cluster motif enrichment\n(log2 corrected OR)",fontsize=5.5,labelpad=1.5); axes.append(cax)
    return axes


def panel_d(fig,spec,ga,gtab,clusters):
    sub=spec.subgridspec(1,3,width_ratios=[.43,1,.045],wspace=.03); lab=fig.add_subplot(sub[0]); ax=fig.add_subplot(sub[1]); cax=fig.add_subplot(sub[2])
    name_to_j={n:i for i,n in enumerate(gtab.gene_name)}; means=[]
    for gene in HEATMAP_GENES: means.append([np.asarray(ga[clusters==c,name_to_j[gene]].mean()).item() for c in CLUSTERS])
    m=np.asarray(means); sd=m.std(1,ddof=0); z=(m-m.mean(1,keepdims=True))/np.where(sd[:,None]>0,sd[:,None],1); vmax=max(2,float(np.nanmax(abs(z))))
    im=ax.imshow(z,aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax); ax.set_xticks(CLUSTERS,[f"C{x}" for x in CLUSTERS]); ax.set_yticks([])
    lab.set_xlim(0,1); lab.set_ylim(len(HEATMAP_GENES)-.5,-.5); lab.axis("off")
    for i,n in enumerate(HEATMAP_GENES): lab.text(.98,i,n,ha="right",va="center",fontstyle="italic",fontsize=6)
    ax.set_title("Cluster-level gene activity",pad=4); fig.colorbar(im,cax=cax).set_label("Scaled gene activity",labelpad=2)
    return [lab,ax,cax]


def panel_e(fig,spec,spatial,ga,gtab):
    sub=spec.subgridspec(2,3,width_ratios=[1,1,.025],wspace=.07,hspace=.08); axes=[]; sc=None
    name_to_j={n:i for i,n in enumerate(gtab.gene_name)}; vmax=2.36
    for i,g in enumerate(SELECTED_GENES):
        ax=fig.add_subplot(sub[i//2,i%2]); val=ga[:,name_to_j[g]].toarray().ravel()
        sc=ax.scatter(spatial.x,spatial.y,c=val,s=3,cmap="viridis",vmin=0,vmax=vmax,lw=0,rasterized=True)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(g,fontstyle="italic",pad=1); axes.append(ax)
    slot=sub[:,2].subgridspec(3,1,height_ratios=[.19,.62,.19]); cax=fig.add_subplot(slot[1]); fig.colorbar(sc,cax=cax).set_label("Gene activity",labelpad=2); axes.append(cax)
    title=fig.add_subplot(sub[:,0]); title.axis("off"); title.set_zorder(-1); title.set_title("Selected gene activity",pad=5); axes.append(title)
    return axes


def panel_f(fig,spec):
    sub=spec.subgridspec(1,3,width_ratios=[.075,.30,.625],wspace=.025); cat=fig.add_subplot(sub[0]); terms=fig.add_subplot(sub[1]); ax=fig.add_subplot(sub[2])
    n=len(ENRICHMENT); ys=np.arange(n)[::-1]; scores=np.asarray([r[3] for r in ENRICHMENT]); colors=[CATEGORY_COLORS[r[0]] for r in ENRICHMENT]
    ax.barh(ys,scores,height=.58,color=colors,edgecolor="none",zorder=2); ax.set_yticks([]); ax.set_xlim(0,100); ax.set_xticks([0,20,40,60,80]); ax.set_ylim(-.65,n-.35)
    ax.tick_params(axis="x",labelsize=6.6,length=2.5,pad=2); ax.set_xlabel("−log10(FDR)",fontsize=7.6,labelpad=3); ax.set_title("Functional enrichment",fontsize=10.5,pad=5,loc="left")
    ax.grid(False,which="both",axis="x")
    for side in ("top","right","left"): ax.spines[side].set_visible(False)
    for y,r in zip(ys,ENRICHMENT): ax.text(r[3]+.8,y,r[4],ha="left",va="center",fontsize=5.55,color="#333",fontstyle="italic",clip_on=True)
    terms.set_xlim(0,1); terms.set_ylim(ax.get_ylim()); terms.axis("off")
    for y,r in zip(ys,ENRICHMENT): terms.text(.98,y,textwrap.fill(r[2],25),ha="right",va="center",fontsize=6.8,color="#222",linespacing=.92)
    cat.set_xlim(0,1); cat.set_ylim(ax.get_ylim()); cat.axis("off")
    cats=np.asarray([r[0] for r in ENRICHMENT])
    for category in ("BP","CC","MF","KEGG"):
        gy=ys[cats==category]; lo,hi=min(gy)-.42,max(gy)+.42
        cat.add_patch(mpl.patches.FancyBboxPatch((.08,lo),.84,hi-lo,boxstyle="round,pad=0.01,rounding_size=0.04",facecolor=CATEGORY_COLORS[category],edgecolor="none"))
        cat.text(.5,(lo+hi)/2,category,color="white",ha="center",va="center",rotation=90,fontweight="bold",fontsize=7.2)
    return [cat,terms,ax]


def main():
    require_inputs(); spatial=load_spatial(); gt_order,gt_colors,methods=matched_methods(spatial)
    a=ad.read_h5ad(ATAC); clusters=spatial.set_index("spot_id").loc[a.obs_names,"SpaCRAFT"].to_numpy(dtype=int)
    ga,gtab=gene_activity(a); marker=marker_matrix(a,clusters); pwms=parse_meme()
    # Numerical identity checks against values recorded in the final-history audit.
    expected={"Nfix":[.987853497,1.319650988,1.168064419,2.069941561,.951961937,.814438284,1.333718856],
              "Foxg1":[.334570874,.315860211,.482820390,1.253401911,1.407703909,.247410642,.245424813],
              "Zfhx2os":[.408696882,.304341581,.484867797,1.379069884,1.074149033,.244898912,.288159914],
              "Sox1":[.402829697,.241236109,.398928750,.675492621,1.439461165,.203561664,.161898829]}
    name_to_j={n:i for i,n in enumerate(gtab.gene_name)}
    for g,want in expected.items():
        got=[np.asarray(ga[clusters==c,name_to_j[g]].mean()).item() for c in CLUSTERS]
        if not np.allclose(got,want,atol=5e-9): raise ValueError(f"{g} gene-activity provenance check failed")

    fig=plt.figure(figsize=(14,9.48),facecolor="white")
    master=fig.add_gridspec(2,1,left=.012,right=.995,bottom=.014,top=.992,hspace=.13,height_ratios=[.94,1.08])
    top=master[0].subgridspec(1,3,width_ratios=[1.29,.81,1.08],wspace=.105)
    bottom=master[1].subgridspec(1,3,width_ratios=[1.08,.92,1.13],wspace=.055)
    groups=[panel_a(fig,top[0],spatial,gt_order,gt_colors,methods),panel_b(fig,top[1],marker),panel_c(fig,top[2],spatial,pwms),
            panel_d(fig,bottom[0],ga,gtab,clusters),panel_e(fig,bottom[1],spatial,ga,gtab),panel_f(fig,bottom[2])]
    fig.canvas.draw()
    for letter,axes in zip("abcdef",groups):
        boxes=[x.get_position() for x in axes]; fig.text(min(x.x0 for x in boxes)-.018,max(x.y1 for x in boxes)+.008,letter,fontsize=13,fontweight="bold",ha="left",va="bottom")
    fig.savefig(PNG,dpi=450,bbox_inches="tight",pad_inches=.025,facecolor="white")
    fig.savefig(PDF,bbox_inches="tight",pad_inches=.025,facecolor="white")
    plt.close(fig)
    print(f"PNG={PNG}\nPDF={PDF}")


if __name__=="__main__": main()

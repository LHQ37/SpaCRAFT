#!/usr/bin/env python3
"""Candidate-only Phase-4 plots; fixed global scale per spatial gene."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/gene_activity_phase4_v1"; FIG=ROOT/"figures/candidates/gene_activity_phase4_v1"
SP=OUT/"spatial_candidates"; FSP=FIG/"spatial_candidates"
for d in (FIG,FSP): d.mkdir(parents=True,exist_ok=True)
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
LABEL=["E12.5-1","E12.5-2","E13.5-1","E13.5-2","E15.5-1","E15.5-2"]

def save(fig,stem):
    for ext in ("png","pdf","svg"): fig.savefig(str(stem)+f".{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
def heat(c):
    r=pd.read_csv(OUT/f"C{c}/C{c}_ranked_gene_activity_dynamics.csv")
    parts=[]
    for p,n in [("EarlyHigh",10),("MiddleHigh",5),("LateHigh",10)]:
        parts.append(r[r.pattern.eq(p)].sort_values("rank_score").head(n))
    z=pd.concat(parts); data=z[ORDER].to_numpy(); data=(data-data.mean(1,keepdims=True))/np.where(data.std(1,keepdims=True)>0,data.std(1,keepdims=True),1)
    pd.DataFrame(data,index=z.gene_symbol,columns=LABEL).to_csv(OUT/f"C{c}/C{c}_gene_activity_heatmap_data.csv")
    fig,ax=plt.subplots(figsize=(7,max(3,.23*len(z)))); sns.heatmap(data,ax=ax,cmap="vlag",center=0,xticklabels=LABEL,yticklabels=z.gene_symbol)
    ax.set_title(f"C{c} descriptive gene activity"); ax.set_xlabel("slice"); ax.set_ylabel("")
    for tick in ax.get_yticklabels(): tick.set_fontstyle("italic")
    cuts=np.cumsum([len(x) for x in parts])[:-1]
    for cut in cuts: ax.axhline(cut,color="black",lw=1.2)
    ax.axvline(2,color="white",lw=2); ax.axvline(4,color="white",lw=2)
    save(fig,FIG/f"C{c}_gene_activity_dynamics_heatmap")
for c in (0,5,6): heat(c)

r=pd.read_csv(OUT/"C5/C5_ranked_gene_activity_dynamics.csv")
top=pd.concat([r[r.pattern.eq(p)].sort_values("rank_score").head(3) for p in ("EarlyHigh","LateHigh")])
top[["gene_id","gene_symbol","pattern",*ORDER]].to_csv(OUT/"C5/C5_top_gene_profiles_data.csv",index=False)
fig,axs=plt.subplots(2,3,figsize=(11,6),sharex=True)
for ax,(_,g) in zip(axs.ravel(),top.iterrows()):
    y=g[ORDER].astype(float).to_numpy(); ax.scatter(range(6),y,color="0.35",zorder=3,label="slice")
    means=np.array([y[:2].mean(),y[2:4].mean(),y[4:].mean()]); lo=np.array([y[:2].min(),y[2:4].min(),y[4:].min()]); hi=np.array([y[:2].max(),y[2:4].max(),y[4:].max()])
    ax.errorbar([.5,2.5,4.5],means,yerr=np.vstack([means-lo,hi-means]),fmt="-o",color="#b2182b",capsize=3,label="stage mean/range")
    ax.set_title(f"{g.gene_symbol} ({g.pattern})",fontstyle="italic"); ax.set_xticks(range(6),LABEL,rotation=45,ha="right")
    ax.set_ylabel("log2(CPM+1)")
axs.ravel()[0].legend(fontsize=7)
save(fig,FIG/"C5_top_gene_activity_profiles")

# Primary ranks 1-2 and backup ranks 3-4, capped by availability.
adn=ad.read_h5ad(OUT/"matrices/six_slice_gene_activity_normalized.h5ad")
sets={}
for name,a,b in [("primary",0,2),("backup",2,4)]:
    genes=pd.concat([r[r.pattern.eq(p)].sort_values("rank_score").iloc[a:b] for p in ("EarlyHigh","LateHigh")])
    sets[name]=genes
    genes[["gene_id","gene_symbol","pattern","rank_score"]].to_csv(SP/f"C5_{name}_spatial_genes.csv",index=False)
    n=len(genes); fig,axs=plt.subplots(max(1,n),6,figsize=(15,2.5*max(1,n)),squeeze=False)
    for row,(_,g) in enumerate(genes.iterrows()):
        j=adn.var_names.get_loc(g.gene_id); val=np.asarray(adn.X[:,j].toarray()).ravel(); vmin,vmax=float(val.min()),float(val.max())
        dat=pd.DataFrame({"obs_name":adn.obs_names,"section":adn.obs.section,"x":adn.obsm["spatial"][:,0],"y":adn.obsm["spatial"][:,1],
                          "gene_id":g.gene_id,"gene_symbol":g.gene_symbol,"value":val,"vmin":vmin,"vmax":vmax})
        dat.to_csv(SP/f"{name}_{g.gene_symbol}_spatial_plot_data.csv.gz",index=False,compression="gzip")
        onefig,oneaxs=plt.subplots(2,3,figsize=(10,6))
        for col,s in enumerate(ORDER):
            m=adn.obs.section.eq(s).to_numpy()
            for ax in (axs[row,col],oneaxs.ravel()[col]):
                ax.scatter(adn.obsm["spatial"][m,0],adn.obsm["spatial"][m,1],c=val[m],s=4,cmap="viridis",vmin=vmin,vmax=vmax,rasterized=True)
                ax.set_aspect("equal"); ax.set_title(LABEL[col]); ax.set_xticks([]); ax.set_yticks([])
        onefig.suptitle(f"{g.gene_symbol} — {g.pattern}; global gene scale")
        save(onefig,FSP/f"C5_{name}_{g.gene_symbol}_spatial")
    fig.suptitle(f"C5 {name} spatial candidates; one global scale per gene")
    save(fig,FSP/f"C5_{name}_gene_spatial_maps")

s=pd.read_csv(OUT/"module_concordance/C5_module_gene_activity_concordance.csv")
fig,ax=plt.subplots(figsize=(6,4)); xx=np.arange(len(s)); ax.bar(xx,s.concordance_fraction_among_tested,label="concordant"); ax.bar(xx,s.opposite_fraction_among_tested,bottom=s.concordance_fraction_among_tested,label="opposite")
ax.set_xticks(xx,s.module); ax.set_ylim(0,1); ax.set_ylabel("fraction among tested genes"); ax.legend(); save(fig,FIG/"C5_module_gene_activity_concordance")

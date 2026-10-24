#!/usr/bin/env python3
"""Candidate-only motif figures from frozen deterministic ranks."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/motif_phase5_v1"
FIG=ROOT/"figures/candidates/motif_phase5_v1"; FIG.mkdir(parents=True,exist_ok=True)
COMP=OUT/"comparison"; SENS=OUT/"sensitivity"
nr={m:pd.read_csv(OUT/m/"nonredundant_motif_enrichment_finite.csv") for m in ("EarlyHigh","LateHigh")}
chosen=[]
for m,x in nr.items():
 chosen.extend(x.sort_values(["enriched","BH_adjusted_pvalue","log2_corrected_odds_ratio","motif_id"],ascending=[False,True,False,True]).head(8).motif_id)
ids=list(dict.fromkeys(chosen))[:16]
cmp=pd.read_csv(COMP/"EarlyHigh_vs_LateHigh_motif_comparison.csv").set_index("motif_id")
sel=cmp.loc[ids].copy()
mat=sel[["EarlyHigh_log2OR","LateHigh_log2OR"]]; mat.columns=["Early-high","Late-high"]
plot=sel.reset_index()[["motif_id","TF_symbol","motif_family","EarlyHigh_log2OR","EarlyHigh_FDR","LateHigh_log2OR","LateHigh_FDR",
 "EarlyHigh_foreground_peak_count","EarlyHigh_foreground_total","LateHigh_foreground_peak_count","LateHigh_foreground_total"]]
plot.to_csv(COMP/"C5_module_motif_display_data.csv",index=False)
labels=sel.TF_symbol
def save(fig,name):
 for e in ("png","pdf","svg"): fig.savefig(FIG/f"{name}.{e}",dpi=220,bbox_inches="tight",facecolor="white")
 plt.close(fig)
fig,ax=plt.subplots(figsize=(5,max(5,.33*len(sel)))); sns.heatmap(mat,ax=ax,cmap="vlag",center=0,yticklabels=labels)
ax.set_ylabel("TF motif / family"); ax.set_title("C5 module motif over-representation")
for i,mid in enumerate(ids):
 for j,fdr in enumerate([sel.loc[mid,"EarlyHigh_FDR"],sel.loc[mid,"LateHigh_FDR"]]):
  if fdr<=.05: ax.scatter(j+.5,i+.5,s=18,c="black")
save(fig,"C5_module_motif_enrichment_heatmap")

long=[]
for mid,z in sel.iterrows():
 for j,m in enumerate(("EarlyHigh","LateHigh")):
  long.append({"motif_id":mid,"TF_symbol":z.TF_symbol,"module":m,"x":j,"log2OR":z[f"{m}_log2OR"],
   "FDR":z[f"{m}_FDR"],"proportion":z[f"{m}_foreground_peak_count"]/z[f"{m}_foreground_total"]})
long=pd.DataFrame(long)
fig,ax=plt.subplots(figsize=(6,max(5,.33*len(sel))))
sc=ax.scatter(long.x,[list(labels).index(v) for v in long.TF_symbol],s=40+260*long.proportion,c=long.log2OR,cmap="vlag",vmin=-abs(long.log2OR).max(),vmax=abs(long.log2OR).max(),edgecolors=np.where(long.FDR<=.05,"black","none"))
ax.set_xticks([0,1],["Early-high","Late-high"]); ax.set_yticks(range(len(labels)),labels); fig.colorbar(sc,ax=ax,label="log2 corrected odds ratio")
ax.set_title("C5 module motif over-representation"); save(fig,"C5_module_motif_enrichment_dotplot")

x=pd.read_csv(COMP/"EarlyHigh_vs_LateHigh_motif_comparison.csv")
fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(x.EarlyHigh_log2OR,x.LateHigh_log2OR,s=12,alpha=.45); lo=min(x.EarlyHigh_log2OR.min(),x.LateHigh_log2OR.min()); hi=max(x.EarlyHigh_log2OR.max(),x.LateHigh_log2OR.max()); ax.plot([lo,hi],[lo,hi],"--",c="0.5")
for z in x.sort_values("display_rank").head(12).itertuples(): ax.annotate(z.TF_symbol,(z.EarlyHigh_log2OR,z.LateHigh_log2OR),fontsize=7)
ax.set_xlabel("EarlyHigh log2 corrected OR"); ax.set_ylabel("LateHigh log2 corrected OR"); ax.set_title("C5 Early/Late motif contrast")
save(fig,"C5_EarlyHigh_LateHigh_motif_contrast")

s=pd.read_csv(SENS/"MOTIF_BACKGROUND_SENSITIVITY.csv")
fig,axs=plt.subplots(1,3,figsize=(10,3.5))
for ax,col,title in zip(axs,["effect_spearman","top20_overlap_fraction","direction_concordance"],["effect rank correlation","top-20 overlap","direction concordance"]):
 ax.bar(s.module,s[col]); ax.set_ylim(0,1); ax.set_title(title)
save(fig,"C5_motif_background_sensitivity")

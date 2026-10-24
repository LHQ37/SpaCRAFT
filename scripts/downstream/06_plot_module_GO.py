#!/usr/bin/env python3
"""Honest candidate/QC plots, including explicit empty enrichment panels."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/GO_phase6_v1"
FIG=ROOT/"figures/candidates/GO_phase6_v1"; FIG.mkdir(parents=True,exist_ok=True)
COMP=OUT/"comparison"; MAP=OUT/"gene_mapping"
nr={m:pd.read_csv(OUT/m/"nonredundant_GO_BP.csv") for m in ("EarlyHigh","LateHigh")}
def save(fig,name):
 for e in ("png","pdf","svg"): fig.savefig(FIG/f"{name}.{e}",dpi=220,bbox_inches="tight",facecolor="white")
 plt.close(fig)
selected=[]
for m in ("EarlyHigh","LateHigh"):
 if len(nr[m]): selected.extend(nr[m].sort_values(["BH_adjusted_pvalue","foreground_gene_count","GO_ID"],ascending=[True,False,True]).head(5).GO_ID)
ids=list(dict.fromkeys(selected))[:12]; comp=pd.read_csv(COMP/"EarlyHigh_vs_LateHigh_GO_comparison.csv").set_index("GO_ID")
if ids:
 sel=comp.loc[ids]; labels=[textwrap.fill(t,42) for t in sel.term]
 rows=[]
 for mid,z in sel.iterrows():
  for j,m in enumerate(("EarlyHigh","LateHigh")):
   rows.append({"GO_ID":mid,"term":z.term,"module":m,"x":j,"log2OR":z[f"{m}_log2OR"],"FDR":z[f"{m}_FDR"],
    "gene_count":z[f"{m}_gene_count"]})
 long=pd.DataFrame(rows); long.to_csv(COMP/"C5_module_GO_BP_display_data.csv",index=False)
 fig,ax=plt.subplots(figsize=(8,max(5,.45*len(sel))))
 yy=[ids.index(v) for v in long.GO_ID]; mm=max(1,abs(long.log2OR).max())
 sc=ax.scatter(long.x,yy,s=35+35*long.gene_count,c=long.log2OR,cmap="vlag",vmin=-mm,vmax=mm,edgecolors=np.where(long.FDR<=.05,"black","none"))
 ax.set_xticks([0,1],["Early-high","Late-high"]); ax.set_yticks(range(len(ids)),labels); fig.colorbar(sc,ax=ax,label="log2 corrected odds ratio")
 ax.set_title("C5 module GO BP enrichment"); save(fig,"C5_module_GO_BP_dotplot")
 mat=sel[["EarlyHigh_log2OR","LateHigh_log2OR"]].to_numpy()
 fig,ax=plt.subplots(figsize=(6,max(5,.4*len(sel)))); im=ax.imshow(mat,aspect="auto",cmap="vlag",vmin=-mm,vmax=mm)
 ax.set_xticks([0,1],["Early-high","Late-high"]); ax.set_yticks(range(len(sel)),labels); fig.colorbar(im,ax=ax,label="log2 corrected odds ratio")
 save(fig,"C5_module_GO_BP_heatmap")
else:
 pd.DataFrame(columns=["GO_ID","term","module","log2OR","FDR","gene_count"]).to_csv(COMP/"C5_module_GO_BP_display_data.csv",index=False)
 for name,title in [("C5_module_GO_BP_dotplot","GO BP dot plot"),("C5_module_GO_BP_heatmap","GO BP heatmap")]:
  fig,ax=plt.subplots(figsize=(7,3.8)); ax.text(.5,.55,"No GO BP term passed the frozen BH FDR < 0.05 rule",ha="center",va="center",transform=ax.transAxes)
  ax.text(.5,.4,"EarlyHigh and LateHigh: INSUFFICIENT_ENRICHMENT",ha="center",va="center",transform=ax.transAxes,fontsize=9,color="0.35")
  ax.set_title(title); ax.axis("off"); save(fig,name)

x=comp.reset_index()
fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(x.EarlyHigh_log2OR,x.LateHigh_log2OR,s=8,alpha=.25)
lo=min(x.EarlyHigh_log2OR.min(),x.LateHigh_log2OR.min()); hi=max(x.EarlyHigh_log2OR.max(),x.LateHigh_log2OR.max()); ax.plot([lo,hi],[lo,hi],"--",color=".5")
for z in x.sort_values("display_rank").head(10).itertuples(): ax.annotate(textwrap.fill(z.term,22),(z.EarlyHigh_log2OR,z.LateHigh_log2OR),fontsize=6)
ax.set_xlabel("EarlyHigh log2 corrected OR"); ax.set_ylabel("LateHigh log2 corrected OR"); ax.set_title("C5 Early/Late GO BP contrast (no enriched term)")
save(fig,"C5_EarlyHigh_LateHigh_GO_contrast")

rows=[]
for m,f in [("EarlyHigh","EarlyHigh_peak_to_gene.csv"),("LateHigh","LateHigh_peak_to_gene.csv")]:
 z=pd.read_csv(MAP/f)
 cat=np.where(z.annotation.eq("intron"),"intronic",np.where(z.annotation.isin(["promoter","5' UTR","exon"]),"promoter/gene body",np.where(z.mapping_rule.eq("NEAREST_GENE_WITHIN_100KB"),"distal/intergenic nearest-gene","unmapped")))
 for c,n in pd.Series(cat).value_counts().items(): rows.append({"module":m,"mapping_category":c,"peak_count":n,"proportion":n/len(z)})
d=pd.DataFrame(rows); d.to_csv(COMP/"C5_module_peak_gene_mapping_composition_data.csv",index=False)
pv=d.pivot(index="module",columns="mapping_category",values="proportion").fillna(0)
fig,ax=plt.subplots(figsize=(7,4)); pv.plot.bar(stacked=True,ax=ax); ax.set_ylabel("peak proportion"); ax.set_xlabel(""); ax.legend(bbox_to_anchor=(1.02,1),loc="upper left")
save(fig,"C5_module_peak_gene_mapping_composition")

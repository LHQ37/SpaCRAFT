#!/usr/bin/env python3
"""Spatial candidates with one global color scale per module across six slices."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/module_spatial_phase7_v1"
FIG=ROOT/"figures/candidates/module_spatial_phase7_v1"; FIG.mkdir(parents=True,exist_ok=True)
SCO=OUT/"scores"; ST=OUT/"stage_summary"; META=OUT/"metadata"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGES=["E12.5","E12.5","E13.5","E13.5","E15.5","E15.5"]
x=pd.read_csv(SCO/"C5_spot_module_scores.csv.gz"); c=x[x.is_C5.astype(bool)]
limits={m:(float(c[f"{m}_module_score"].min()),float(c[f"{m}_module_score"].max())) for m in ("EarlyHigh","LateHigh")}
(META/"spatial_global_color_limits.json").write_text(__import__("json").dumps(limits,indent=2)+"\n")
def save(fig,name):
 for e in ("png","pdf","svg"): fig.savefig(FIG/f"{name}.{e}",dpi=220,bbox_inches="tight",facecolor="white")
 plt.close(fig)
def spatial(figsize,name,grouped=False):
 fig,axs=plt.subplots(2,6,figsize=figsize)
 last={}
 for r,m in enumerate(("EarlyHigh","LateHigh")):
  vmin,vmax=limits[m]
  for j,s in enumerate(ORDER):
   ax=axs[r,j]; z=x[x.slice.eq(s)]; non=~z.is_C5.astype(bool); yes=z.is_C5.astype(bool)
   ax.scatter(z.loc[non,"spatial_x"],z.loc[non,"spatial_y"],s=2,c="#dedede",rasterized=True)
   last[m]=ax.scatter(z.loc[yes,"spatial_x"],z.loc[yes,"spatial_y"],s=4,c=z.loc[yes,f"{m}_module_score"],cmap="viridis",vmin=vmin,vmax=vmax,rasterized=True)
   ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
   ax.set_title(f"{STAGES[j]}\n{s}",fontsize=8)
   if j==0: ax.set_ylabel(f"{m}\nmodule score",fontsize=9)
  fig.colorbar(last[m],ax=axs[r,:].tolist(),fraction=.012,pad=.01,label=f"{m} global score")
 if grouped:
  for cut in (2,4):
   xx=(axs[0,cut-1].get_position().x1+axs[0,cut].get_position().x0)/2
   fig.add_artist(plt.Line2D([xx,xx],[.08,.94],transform=fig.transFigure,color=".65",lw=1))
 fig.suptitle("Frozen C5 accessibility-module scores; non-C5 tissue outline in grey",fontsize=12)
 save(fig,name)
spatial((18,6),"C5_EarlyHigh_LateHigh_module_spatial_maps")
spatial((14,4.6),"C5_module_spatial_compact_candidate_A")
spatial((15,5.2),"C5_module_spatial_compact_candidate_B",grouped=True)

d=pd.read_csv(ST/"C5_module_score_by_slice.csv")
fig,ax=plt.subplots(figsize=(7,4.5))
colors={"EarlyHigh":"#2166ac","LateHigh":"#b2182b"}
for m in ("EarlyHigh","LateHigh"):
 z=d[d.module.eq(m)].copy(); z["stage_x"]=z.stage.map({"E12.5":0,"E13.5":1,"E15.5":2})
 jitter=np.tile([-.07,.07],3); ax.scatter(z.stage_x+jitter,z["mean"],color=colors[m],s=35,label=f"{m} slices")
 sm=z.groupby("stage_x")["mean"].mean().reindex([0,1,2]); ax.plot([0,1,2],sm,color=colors[m],marker="o",lw=2,label=f"{m} stage mean")
ax.set_xticks([0,1,2],["E12.5","E13.5","E15.5"]); ax.set_ylabel("C5 bias-controlled module score")
ax.set_title("Descriptive C5 module-score summary"); ax.legend(ncol=2,fontsize=8); save(fig,"C5_spatial_module_score_stage_summary")

#!/usr/bin/env python3
"""Plot fixed C5 accessibility modules and stability candidates."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/accessibility_phase3_v1"; FIG=ROOT/"figures/candidates/accessibility_phase3_v1"
FIG.mkdir(parents=True,exist_ok=True)
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGE=["E12.5","E12.5","E13.5","E13.5","E15.5","E15.5"]
SCOL={"E12.5":"#4E79A7","E13.5":"#F28E2B","E15.5":"#59A14F"}
MCOL={"Module_EarlyHigh":"#3B6FB6","Module_LateHigh":"#D95F4F"}
mem=pd.read_csv(OUT/"modules/C5_all_modules_membership.csv").sort_values(["module","phase2_rank"])
zdf=pd.read_csv(OUT/"matrices/C5_modules_row_zscore_six_slices.csv.gz").set_index("peak_id")
z=zdf.loc[mem.peak_id,ORDER].to_numpy()
split=(mem.module=="Module_EarlyHigh").sum()
profiles=pd.read_csv(OUT/"profiles/C5_module_profiles_by_slice.csv")
stageprof=pd.read_csv(OUT/"profiles/C5_module_profiles_by_stage_descriptive.csv")
loo=pd.read_csv(OUT/"stability/C5_module_leave_one_slice_out.csv")
same=pd.read_csv(OUT/"stability/C5_module_same_stage_consistency.csv")

def save3(fig,name):
    for ext in ("png","pdf","svg"): fig.savefig(FIG/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
def draw_heat(ax,with_colorbar=True):
    im=ax.imshow(z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5)
    ax.axhline(split-.5,color="black",lw=2)
    ax.set(xticks=range(6),xticklabels=[f"{s}\n{st}" for s,st in zip(ORDER,STAGE)],
           yticks=[(split-1)/2,split+(len(mem)-split-1)/2],
           yticklabels=[f"Early-high\n(n={split})",f"Late-high\n(n={len(mem)-split})"],
           ylabel="Fixed Phase 2 module membership",xlabel="Six slices retained separately")
    ax.tick_params(axis="x",rotation=25,labelsize=8)
    ax.set_title("C5 accessibility modules (232 peaks)")
    if with_colorbar: plt.colorbar(im,ax=ax,label="Row z-score of log2(CPM+1)",fraction=.035,pad=.02)
    return im
def draw_profile(ax):
    xs=np.arange(3); stages=["E12.5","E13.5","E15.5"]
    for module in ("Module_EarlyHigh","Module_LateHigh"):
        d=stageprof[stageprof.module.eq(module)].set_index("stage").loc[stages]
        means=d.stage_mean_row_zscore.to_numpy()
        ax.plot(xs,means,color=MCOL[module],marker="o",lw=2,
                label=module.replace("Module_","").replace("High","-high"))
        for i,(_,r) in enumerate(d.iterrows()):
            vals=[r.slice_1_mean_row_zscore,r.slice_2_mean_row_zscore]
            ax.vlines(i,min(vals),max(vals),color=MCOL[module],lw=2,alpha=.65)
            ax.scatter([i-.045,i+.045],vals,s=24,facecolor="white",edgecolor=MCOL[module],zorder=3)
    ax.axhline(0,color="grey",lw=.8)
    ax.set(xticks=xs,xticklabels=stages,ylabel="Module mean row z-score",
           xlabel="Stage (two slice points shown)",title="Module centroid profiles")
    ax.legend(frameon=False)

fig,ax=plt.subplots(figsize=(8,9)); draw_heat(ax); fig.suptitle("Stage-associated accessibility modules",y=.995)
save3(fig,"C5_full_accessibility_modules_heatmap")
fig,ax=plt.subplots(figsize=(7,5)); draw_profile(ax); fig.suptitle("Stage-associated accessibility modules",y=1.02)
save3(fig,"C5_module_centroid_profiles")
fig=plt.figure(figsize=(13,8)); gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=.35)
ax1=fig.add_subplot(gs[0,0]); im=draw_heat(ax1,False)
ax2=fig.add_subplot(gs[0,1]); draw_profile(ax2)
fig.colorbar(im,ax=ax1,label="Row z-score",fraction=.035,pad=.02)
fig.suptitle("Stage-associated accessibility modules",fontsize=16)
save3(fig,"C5_accessibility_modules_composite")

fig,axs=plt.subplots(1,2,figsize=(11,4.5))
pv=loo.pivot(index="module",columns="omitted_slice",values="expected_direction_retained").loc[
    ["Module_EarlyHigh","Module_LateHigh"],ORDER].astype(int)
axs[0].imshow(pv,cmap=ListedColormap(["#D95F4F","#59A14F"]),vmin=0,vmax=1,aspect="auto")
axs[0].set(xticks=range(6),xticklabels=ORDER,yticks=range(2),
           yticklabels=["Early-high","Late-high"],title="Leave-one-slice-out direction retained")
axs[0].tick_params(axis="x",rotation=30)
for i in range(2):
    for j in range(6): axs[0].text(j,i,"Yes" if pv.iloc[i,j] else "No",ha="center",va="center")
sd=same.pivot(index="module",columns="stage",values="absolute_module_score_difference").loc[
    ["Module_EarlyHigh","Module_LateHigh"],["E12.5","E13.5","E15.5"]]
x=np.arange(3); w=.34
for i,module in enumerate(sd.index):
    axs[1].bar(x+(i-.5)*w,sd.loc[module],width=w,color=MCOL[module],
               label=module.replace("Module_","").replace("High","-high"))
axs[1].axhline(.5,color="grey",ls="--",lw=1,label="preregistered small-difference threshold")
axs[1].set(xticks=x,xticklabels=sd.columns,ylabel="Absolute module-score difference",
           title="Same-stage slice differences")
axs[1].legend(frameon=False,fontsize=8)
fig.suptitle("C5 module stability audit — descriptive only")
fig.tight_layout()
fig.savefig(FIG/"C5_module_stability_summary.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("logical_candidate_figures=4")

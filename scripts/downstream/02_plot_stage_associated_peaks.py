#!/usr/bin/env python3
"""Generate Phase 2 candidate figures from preregistered descriptive results."""
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

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/accessibility_phase2_v1"; PB=OUT/"pseudobulk"; QC=OUT/"qc"
FIG=ROOT/"figures/candidates/accessibility_phase2_v1"; FIG.mkdir(parents=True,exist_ok=True)
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGE=["E12.5","E12.5","E13.5","E13.5","E15.5","E15.5"]
COL={"early-high":"#3B6FB6","middle-high":"#8E63A9","late-high":"#D95F4F"}

def save3(fig,name):
    for ext in ("png","pdf","svg"): fig.savefig(FIG/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)

def selected_rows(cluster):
    r=pd.read_csv(OUT/f"C{cluster}/C{cluster}_ranked_stage_associated_peaks.csv")
    picks=[]
    for pat,n in [("early-high",15),("middle-high",10),("late-high",15)]:
        x=r[r.pattern.eq(pat)].sort_values("rank").head(n).copy()
        picks.append(x)
    return pd.concat(picks,ignore_index=True) if picks else r.head(0)

heat_data={}
for cluster in (5,6):
    meta=pd.read_csv(PB/"pseudobulk_sample_metadata.csv").query("cluster==@cluster")
    logdf=pd.read_csv(PB/f"C{cluster}_log2CPM.csv.gz")
    tested=pd.read_csv(OUT/f"C{cluster}/C{cluster}_tested_peaks.csv").query("tested")
    x=logdf.set_index("peak_id").loc[tested.peak_id,ORDER]
    corr=x.corr()
    cons=pd.read_csv(QC/f"C{cluster}_same_stage_peak_consistency.csv")
    fig,axs=plt.subplots(2,2,figsize=(11,9))
    axs[0,0].bar(range(6),meta.set_index("slice").loc[ORDER].library_size,color="#4E79A7")
    axs[0,0].set(xticks=range(6),xticklabels=ORDER,ylabel="Library size",title="Pseudobulk library size")
    axs[0,0].tick_params(axis="x",rotation=30)
    axs[0,1].bar(range(6),meta.set_index("slice").loc[ORDER].detected_peak_count,color="#59A14F")
    axs[0,1].set(xticks=range(6),xticklabels=ORDER,ylabel="Detected peaks",title="Detected peak count")
    axs[0,1].tick_params(axis="x",rotation=30)
    im=axs[1,0].imshow(corr,vmin=0,vmax=1,cmap="viridis")
    axs[1,0].set(xticks=range(6),xticklabels=ORDER,yticks=range(6),yticklabels=ORDER,
                 title="Tested-peak log2CPM correlation")
    axs[1,0].tick_params(axis="x",rotation=30)
    for i in range(6):
        for j in range(6): axs[1,0].text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",
                                         color="white" if corr.iloc[i,j]<.45 else "black",fontsize=7)
    fig.colorbar(im,ax=axs[1,0],fraction=.045)
    for st,(i,j),color in zip(["E12.5","E13.5","E15.5"],[(0,1),(2,3),(4,5)],
                              ["#4E79A7","#F28E2B","#59A14F"]):
        axs[1,1].scatter(x.iloc[:,i],x.iloc[:,j],s=3,alpha=.15,color=color,label=st)
    lim=[0,float(x.to_numpy().max())]
    axs[1,1].plot(lim,lim,"--",color="grey",lw=1)
    axs[1,1].set(xlabel="Slice 1 log2(CPM+1)",ylabel="Slice 2 log2(CPM+1)",
                 title="Same-stage slice pairs")
    axs[1,1].legend(frameon=False)
    fig.suptitle(f"C{cluster} pseudobulk QC — descriptive only")
    fig.tight_layout()
    fig.savefig(FIG/f"C{cluster}_pseudobulk_QC.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    sel=selected_rows(cluster)
    if len(sel):
        vals=logdf.set_index("peak_id").loc[sel.peak_id,ORDER].to_numpy()
        z=(vals-vals.mean(axis=1,keepdims=True))/np.where(vals.std(axis=1,keepdims=True)==0,1,
                                                          vals.std(axis=1,keepdims=True))
        pdata=sel[["peak_id","pattern","rank"]].reset_index(drop=True)
        for j,s in enumerate(ORDER): pdata[s+"_row_zscore"]=z[:,j]
        pdata.to_csv(OUT/f"C{cluster}/C{cluster}_stage_associated_peak_heatmap_data.csv",index=False)
        heat_data[cluster]=(pdata,z)
        fig,ax=plt.subplots(figsize=(9,max(5,len(sel)*.25)))
        im=ax.imshow(z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5)
        ax.set(xticks=range(6),xticklabels=[f"{s}\n{st}" for s,st in zip(ORDER,STAGE)],
               yticks=range(len(sel)),yticklabels=sel.peak_id, title=f"C{cluster} stage-associated peak candidates")
        ax.tick_params(axis="x",rotation=25); ax.tick_params(axis="y",labelsize=6)
        ax.set_xlabel("Six slices retained separately"); ax.set_ylabel("Preregistered ranked peaks")
        fig.colorbar(im,ax=ax,label="Row z-score of log2(CPM+1)",fraction=.035,pad=.03)
        save3(fig,f"C{cluster}_stage_associated_peak_heatmap")
    else:
        heat_data[cluster]=(pd.DataFrame(),np.empty((0,6)))

    ranked=pd.read_csv(OUT/f"C{cluster}/C{cluster}_ranked_stage_associated_peaks.csv")
    el=ranked[ranked.pattern.isin(["early-high","late-high"])].sort_values("rank")
    el.to_csv(OUT/f"C{cluster}/C{cluster}_stage_effect_rank_plot_data.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    for pat,g in el.groupby("pattern"):
        ax.scatter(g["rank"],g.early_to_late_delta,s=13,alpha=.75,color=COL[pat],label=pat)
    ax.axhline(0,color="grey",lw=.8)
    ax.set(xlabel="Preregistered candidate rank",ylabel="E15.5 mean − E12.5 mean log2CPM",
           title=f"C{cluster} descriptive stage-effect ranking")
    ax.legend(frameon=False)
    fig.savefig(FIG/f"C{cluster}_stage_effect_rank_plot.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    prof=[]
    for pat in ("early-high","late-high"):
        prof.append(ranked[(ranked.pattern.eq(pat))&ranked.strict_pairwise_concordant].sort_values("rank").head(3))
    prof=pd.concat(prof,ignore_index=True)
    summary=pd.read_csv(OUT/f"C{cluster}/C{cluster}_peak_stage_summary.csv.gz").set_index("peak_id")
    rows=[]
    fig,axs=plt.subplots(2,3,figsize=(11,6),sharex=True)
    for ax,(_,r) in zip(axs.flat,prof.iterrows()):
        q=summary.loc[r.peak_id]; means=[]
        for ii,st in enumerate(("E12.5","E13.5","E15.5")):
            a=q[f"{st}_mean"]; means.append(a)
            vals=[q[f"{ORDER[ii*2]}_log2CPM"],q[f"{ORDER[ii*2+1]}_log2CPM"]]
            ax.vlines(ii,min(vals),max(vals),color=COL[r.pattern],lw=2)
            ax.scatter([ii-.05,ii+.05],vals,color=COL[r.pattern],s=20)
            rows += [{"peak_id":r.peak_id,"pattern":r.pattern,"stage":st,"stage_mean":a,
                      "slice_1_value":vals[0],"slice_2_value":vals[1]}]
        ax.plot(range(3),means,color=COL[r.pattern],marker="o")
        ax.set_title(str(r.peak_id),fontsize=8)
        ax.set_xticks(range(3),["E12.5","E13.5","E15.5"]); ax.set_ylabel("log2(CPM+1)")
    for ax in axs.flat[len(prof):]: ax.axis("off")
    fig.suptitle(f"C{cluster} top strict early/late peak profiles — descriptive slice points")
    fig.tight_layout()
    fig.savefig(FIG/f"C{cluster}_top_peak_stage_profiles.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/f"C{cluster}/C{cluster}_top_peak_stage_profiles_data.csv",index=False)

combined_meta=[]; combined_z=[]
for cluster in (5,6):
    pdata,z=heat_data[cluster]
    if len(pdata):
        combined_meta.append(pdata.assign(cluster=f"C{cluster}"))
        combined_z.append(z)
if combined_z:
    cm=pd.concat(combined_meta,ignore_index=True); cz=np.vstack(combined_z)
    cm.to_csv(OUT/"combined/C5_C6_stage_associated_peak_heatmap_data.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,max(6,len(cm)*.18)))
    im=ax.imshow(cz,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5)
    ax.set(xticks=range(6),xticklabels=[f"{s}\n{st}" for s,st in zip(ORDER,STAGE)],
           yticks=range(len(cm)),yticklabels=[f"{c} | {p}" for c,p in zip(cm.cluster,cm.peak_id)],
           title="C5 and C6 stage-associated peak candidates")
    ax.tick_params(axis="x",rotation=25); ax.tick_params(axis="y",labelsize=5)
    ax.set_xlabel("Six slices retained separately")
    fig.colorbar(im,ax=ax,label="Row z-score of log2(CPM+1)",fraction=.03,pad=.02)
    save3(fig,"C5_C6_stage_associated_peak_heatmap_combined")
print("logical_candidate_figures=9")

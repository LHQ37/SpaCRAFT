#!/usr/bin/env python3
"""Create Phase 1 composition candidate figures; no final Figure 6 assembly."""
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
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT=MULTI_DOWNSTREAM_ROOT
RES=ROOT/"results/composition_v1"
FIG=ROOT/"figures/candidates/composition_v1"
FIG.mkdir(parents=True,exist_ok=True)
palette=pd.read_csv(RES/"metadata/cluster_color_palette.csv").set_index("cluster").color_hex.to_dict()
six_order=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
four_order=["E11_S1","E13_S1","E15_S1","E18_S1"]
stage6=["E12.5","E12.5","E13.5","E13.5","E15.5","E15.5"]

def save3(fig,name):
    for ext in ("png","pdf","svg"):
        fig.savefig(FIG/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)

def stacked(props,order,k,title,name,subtitle=""):
    pivot=props.pivot(index="slice",columns="cluster",values="cluster_proportion_within_slice").loc[order]
    fig,ax=plt.subplots(figsize=(9,5.5))
    bottom=np.zeros(len(order))
    for c in range(k):
        ax.bar(range(len(order)),pivot[c],bottom=bottom,color=palette[c],label=f"C{c}",
               width=.75,edgecolor="white",linewidth=.3)
        bottom+=pivot[c].to_numpy()
    ax.set(xticks=range(len(order)),xticklabels=order,ylim=(0,1),ylabel="Cluster proportion within slice",
           xlabel="Slice")
    ax.set_title(title,pad=38)
    ax.tick_params(axis="x",rotation=25)
    if subtitle: ax.text(.5,1.01,subtitle,transform=ax.transAxes,ha="center",fontsize=9)
    ax.legend(ncol=min(k,8),bbox_to_anchor=(.5,-.25),loc="upper center",frameon=False)
    fig.subplots_adjust(bottom=.28)
    save3(fig,name)

def prop_heatmap(props,order,k,ann,title,name,row_order=None,subtitle=""):
    row_order=list(range(k)) if row_order is None else row_order
    m=props.pivot(index="cluster",columns="slice",values="cluster_proportion_within_slice").loc[row_order,order]
    fig,ax=plt.subplots(figsize=(9,6.5 if k<=11 else 8))
    im=ax.imshow(m,aspect="auto",cmap="Blues",vmin=0,vmax=float(m.to_numpy().max()))
    ax.set(xticks=range(len(order)),xticklabels=order,yticks=range(k),
           yticklabels=[f"C{x} | {ann.loc[x,'majority_cell_type']}" for x in row_order],
           xlabel="Slice",ylabel="Frozen SpaCRAFT cluster")
    ax.set_title(title,pad=38)
    ax.tick_params(axis="x",rotation=25)
    for i in range(k):
        for j in range(len(order)):
            ax.text(j,i,f"{m.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7,
                    color="white" if m.iloc[i,j]>.55*m.to_numpy().max() else "black")
    if subtitle: ax.text(.5,1.02,subtitle,transform=ax.transAxes,ha="center",fontsize=9)
    fig.colorbar(im,ax=ax,label="Proportion within slice",fraction=.035,pad=.03)
    save3(fig,name)

def coverage_heatmap(cov,order,k,title,name,subtitle=""):
    m=cov.pivot(index="cluster",columns="slice",values="spot_count").loc[range(k),order]
    levels=cov.pivot(index="cluster",columns="slice",values="coverage_level").loc[range(k),order]
    fig,ax=plt.subplots(figsize=(9,6.5 if k<=11 else 8))
    im=ax.imshow(np.log10(m+1),aspect="auto",cmap="YlGnBu")
    ax.set(xticks=range(len(order)),xticklabels=order,yticks=range(k),
           yticklabels=[f"C{x}" for x in range(k)],xlabel="Slice",ylabel="Cluster")
    ax.set_title(title,pad=38)
    ax.tick_params(axis="x",rotation=25)
    short={"ADEQUATE":"A","LIMITED":"L","VERY_LOW":"V","ABSENT":"0"}
    for i in range(k):
        for j in range(len(order)):
            ax.text(j,i,f"{int(m.iloc[i,j])}\n{short[levels.iloc[i,j]]}",ha="center",va="center",fontsize=7)
    if subtitle: ax.text(.5,1.02,subtitle,transform=ax.transAxes,ha="center",fontsize=9)
    fig.colorbar(im,ax=ax,label="log10(spot count + 1)",fraction=.035,pad=.03)
    save3(fig,name)

sixp=pd.read_csv(RES/"six_slice/cluster_proportions_by_slice.csv")
fourp=pd.read_csv(RES/"four_slice/cluster_proportions_by_slice.csv")
sixa=pd.read_csv(RES/"six_slice/cluster_celltype_annotation_overall.csv").set_index("cluster")
foura=pd.read_csv(RES/"four_slice/cluster_celltype_annotation_overall.csv").set_index("cluster")
sixcov=pd.read_csv(RES/"six_slice/cluster_slice_coverage.csv")
fourcov=pd.read_csv(RES/"four_slice/cluster_slice_coverage.csv")

stacked(sixp,six_order,11,"Frozen SpaCRAFT cluster composition across six slices",
        "six_slice_cluster_stacked_bar","Stages: E12.5 | E12.5 | E13.5 | E13.5 | E15.5 | E15.5; descriptive only")
prop_heatmap(sixp,six_order,11,sixa,"Cluster proportion by slice",
             "six_slice_cluster_proportion_heatmap_fixed_order",
             subtitle="Fixed slice and cluster order; right labels show majority cell type")

stage=pd.read_csv(RES/"six_slice/cluster_proportions_by_stage_descriptive.csv")
ranges=stage.groupby("cluster").stage_mean.agg(lambda x:x.max()-x.min()).sort_values(ascending=False)
peakstage=stage.loc[stage.groupby("cluster").stage_mean.idxmax()].set_index("cluster").stage
stage_rank={"E12.5":0,"E13.5":1,"E15.5":2}
pattern=sorted(range(11),key=lambda c:(stage_rank[peakstage[c]],-ranges[c],c))
prop_heatmap(sixp,six_order,11,sixa,"Cluster proportion by slice — pattern-order candidate",
             "six_slice_cluster_proportion_heatmap_pattern_order_candidate",row_order=pattern,
             subtitle="Candidate row order only; slice order fixed")

fig,axes=plt.subplots(4,3,figsize=(11,11),sharex=True)
x=np.arange(3)
for c,ax in enumerate(axes.flat):
    if c>=11: ax.axis("off"); continue
    d=stage[stage.cluster.eq(c)].set_index("stage").loc[["E12.5","E13.5","E15.5"]]
    ax.plot(x,d.stage_mean,color=palette[c],marker="o",lw=1.8)
    for i,(_,r) in enumerate(d.iterrows()):
        ax.vlines(i,r.stage_min,r.stage_max,color=palette[c],lw=2,alpha=.65)
        ax.scatter([i-.06,i+.06],[r.slice_1_proportion,r.slice_2_proportion],s=18,
                   facecolor="white",edgecolor=palette[c],zorder=3)
    ax.set_title(f"C{c} | {sixa.loc[c,'majority_cell_type']}",fontsize=9)
    ax.set_xticks(x,["E12.5","E13.5","E15.5"])
    ax.set_ylim(bottom=0)
fig.supylabel("Cluster proportion within slice")
fig.supxlabel("Stage (two independent slice points; descriptive only)")
fig.suptitle("Stage-associated composition summaries for all frozen clusters",y=.995)
fig.tight_layout()
fig.savefig(FIG/"six_slice_cluster_stage_summary_all.png",dpi=300,bbox_inches="tight"); plt.close(fig)

top=ranges.sort_values(ascending=False,kind="mergesort").index[:5].tolist()
fig,ax=plt.subplots(figsize=(8,5.5))
for c in top:
    d=stage[stage.cluster.eq(c)].set_index("stage").loc[["E12.5","E13.5","E15.5"]]
    ax.plot(x,d.stage_mean,color=palette[c],marker="o",label=f"C{c} {sixa.loc[c,'majority_cell_type']}")
    for i,(_,r) in enumerate(d.iterrows()):
        ax.vlines(i,r.stage_min,r.stage_max,color=palette[c],alpha=.55)
        ax.scatter([i-.045,i+.045],[r.slice_1_proportion,r.slice_2_proportion],s=16,color=palette[c])
ax.set(xticks=x,xticklabels=["E12.5","E13.5","E15.5"],ylabel="Cluster proportion within slice",
       xlabel="Stage",title="Top 5 variable clusters by preregistered stage-mean range")
ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
fig.savefig(FIG/"six_slice_cluster_stage_summary_top_variable.png",dpi=300,bbox_inches="tight"); plt.close(fig)

coverage_heatmap(sixcov,six_order,11,"Six-slice frozen-cluster coverage",
                 "six_slice_cluster_coverage_heatmap","A ≥100; L 30–99; V 1–29; 0 absent")
stacked(fourp,four_order,16,"Frozen SpaCRAFT cluster composition across four slices",
        "four_slice_cluster_stacked_bar","Descriptive auxiliary dataset; stage and slice are confounded")
prop_heatmap(fourp,four_order,16,foura,"Four-slice cluster proportion heatmap",
             "four_slice_cluster_proportion_heatmap",
             subtitle="Descriptive auxiliary dataset; cluster IDs are not mapped to six-slice IDs")
coverage_heatmap(fourcov,four_order,16,"Four-slice frozen-cluster coverage",
                 "four_slice_cluster_coverage_heatmap","Descriptive auxiliary dataset; A ≥100; L 30–99; V 1–29; 0 absent")

(RES/"metadata/top_variable_clusters.csv").write_text(
    "rank,cluster,stage_mean_range\n"+"\n".join(f"{i+1},{c},{ranges[c]}" for i,c in enumerate(top))+"\n")
print(f"logical_candidate_figures=9")

#!/usr/bin/env python3
"""Render the formal three-panel SpaCRAFT multi-slice benchmark figure.

Layout: a metrics, b six-slice spatial comparison, c two-row UMAP comparison.
Only frozen result tables, embeddings and cluster labels are read.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import colorsys
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import umap


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FORMAL = REPO / "results/multi_slice"
SOURCE = Path(os.environ.get(
    "SPACRAFT_MULTISLICE_COMPARISON_ROOT",
    REPO / "results/multi_slice",
))
CANONICAL = FORMAL / "canonical/six_slices_obs.csv"
METRICS = FORMAL / "evaluation/six_slices_all_metrics.csv"
OVERALL = FORMAL / "evaluation/CROSS_DATASET_SEVEN_METRIC_BALANCED_RANKING.csv"
PNG = REPO / "figures/main/fig3_multi_slice_benchmark.png"
PDF = REPO / "figures/main/fig3_multi_slice_benchmark.pdf"

METRIC_METHODS = ["INSTINCT","PeakVI","scDREAMER","SEDR","STAligner","GraphST","SpaCRAFT"]
SPATIAL_METHODS = ["SpaCRAFT","INSTINCT","GraphST","PeakVI","scDREAMER","SEDR","STAligner"]
UMAP_METHODS = SPATIAL_METHODS
SLICES = ["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
METHOD_COLORS = {
 "INSTINCT":"#4C9CC9", "PeakVI":"#EEBC4C", "scDREAMER":"#E18E4C",
 "SEDR":"#4CBB9D", "STAligner":"#DBA0C1", "GraphST":"#88CAF0", "SpaCRAFT":"#9758CD",
}
SLICE_COLORS = dict(zip(SLICES,["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00"]))
UMAP_CONFIG = dict(n_neighbors=30,min_dist=.5,metric="euclidean",n_components=2,random_state=1234)

plt.rcParams.update({"font.family":"Liberation Sans","font.size":7,"axes.titlesize":9,
 "axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
 "axes.linewidth":.7,"pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white"})


def method_dir(method):
    return SOURCE / "six_slices" / method


def require_inputs():
    paths=[CANONICAL,METRICS,OVERALL]
    for method in METRIC_METHODS:
        d=method_dir(method); paths += [d/"embedding.npy",d/"obs_names.csv",d/"cluster_labels.csv"]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError("Missing frozen input(s):\n"+"\n".join(missing))


def canonical():
    d=pd.read_csv(CANONICAL).rename(columns={"ground_truth_label":"cell_type"})
    if d.slice_id.drop_duplicates().tolist()!=SLICES: raise ValueError("Unexpected six-slice order")
    return d


def cell_palette(c):
    order=list(dict.fromkeys(c.cell_type))
    cm=plt.get_cmap("tab20",20); cm2=plt.get_cmap("tab20b",20)
    base=[matplotlib.colors.to_rgb(cm(i)) for i in range(20)]+[matplotlib.colors.to_rgb(cm2(i)) for i in range(20)]
    # Deepen the manuscript palette without changing category assignments.
    # Increasing saturation and gently reducing lightness keeps pale tab20
    # entries legible at publication scale while preserving hue identity.
    colors=[]
    for rgb in base:
        h,l,s=colorsys.rgb_to_hls(*rgb)
        colors.append(matplotlib.colors.to_hex(colorsys.hls_to_rgb(h,max(.24,l*.75),min(1,s*1.32))))
    return order,dict(zip(order,colors[:len(order)]))


def aligned_table(method,c):
    d=method_dir(method)
    names=pd.read_csv(d/"obs_names.csv").iloc[:,0].astype(str)
    emb=np.load(d/"embedding.npy")
    if len(names)!=len(emb): raise ValueError(f"{method}: embedding/name mismatch")
    index=pd.Series(np.arange(len(names)),index=names)
    order=index.loc[c.obs_name.astype(str)].to_numpy()
    labels=pd.read_csv(d/"cluster_labels.csv")
    label_col="cluster" if "cluster" in labels else "cluster_label"
    labels=labels.set_index("obs_name").loc[c.obs_name.astype(str),label_col].astype(str).to_numpy()
    return emb[order],labels


def matched_cluster_colors(labels,c,cell_order,palette):
    tab=pd.crosstab(pd.Series(labels,name="cluster"),c.cell_type).reindex(columns=cell_order,fill_value=0)
    rr,cc=linear_sum_assignment(-tab.to_numpy())
    mapping={tab.index[r]:tab.columns[k] for r,k in zip(rr,cc)}
    matched=pd.Series(labels).map(mapping).fillna("Unmatched")
    return matched.map(palette).fillna("#808080").to_numpy(),mapping


def spatial(ax,q,colors,size=1.65):
    ax.scatter(q.spatial_x,q.spatial_y,c=colors,s=size,linewidths=0,rasterized=True)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def legend_handles(palette,order,size=4.6):
    return [Line2D([0],[0],marker="o",linestyle="",markersize=size,
                   markerfacecolor=palette[x],markeredgewidth=0,label=x.replace("_"," ")) for x in order]


def draw_panel_a(fig,spec):
    d=pd.read_csv(METRICS).set_index("method").loc[METRIC_METHODS]
    overall=pd.read_csv(OVERALL).set_index("method").loc[METRIC_METHODS,"Cross_Dataset_Rank_Score_7"]
    items=[("ARI",d.ARI.to_numpy()),("NMI",d.NMI.to_numpy()),
           ("Cell-type ASW",d.Cell_type_ASW.to_numpy()),("kBET",d.kBET.to_numpy()),
           ("Batch ASW",d.Batch_ASW.to_numpy()),("Batch PCR",d.Batch_PCR.to_numpy()),
           ("Graph connectivity",d.Graph_connectivity.to_numpy()),("Overall",overall.to_numpy())]
    # Latest manuscript layout: two rows of four metrics with metric-specific
    # ranges, followed by one compact method legend.
    limits=[
        ((0,.30),[0,.15,.30]), ((0,.50),[0,.25,.50]),
        ((.35,.55),[.35,.45,.55]), ((0,1),[0,.5,1]),
        ((.70,1),[.70,.85,1]), ((.50,1),[.50,.75,1]),
        ((.60,1),[.60,.80,1]), ((.20,.80),[.20,.50,.80]),
    ]
    outer=spec.subgridspec(4,1,height_ratios=[.13,1,1,.16],hspace=.22)
    group_ax=fig.add_subplot(outer[0]); group_ax.set_xlim(0,1); group_ax.set_ylim(0,1); group_ax.axis("off")
    group_ax.text(.30,.54,"Biological conservation",ha="center",va="center",
                  fontsize=8.3,fontweight="semibold",color="#334A46",
                  bbox=dict(boxstyle="round,pad=.22",facecolor="#EAF4F1",edgecolor="none"))
    group_ax.text(.70,.54,"Batch correction",ha="center",va="center",
                  fontsize=8.3,fontweight="semibold",color="#3F4657",
                  bbox=dict(boxstyle="round,pad=.22",facecolor="#EEF1F7",edgecolor="none"))
    grid=outer[1:3].subgridspec(2,4,wspace=.27,hspace=.30); axes=[]
    title_faces=["#EAF4F1"]*3+["#EEF1F7"]*4+["#F2F2F2"]
    for i,(title,values) in enumerate(items):
        ax=fig.add_subplot(grid[i//4,i%4]); bars=ax.bar(np.arange(7),values,color=[METHOD_COLORS[m] for m in METRIC_METHODS],width=.72)
        bars[-1].set_edgecolor("#7A35B5"); bars[-1].set_linewidth(1.15)
        ylim,yticks=limits[i]
        ax.set_title(title,pad=1.5,fontsize=8.2,
                     bbox=dict(boxstyle="round,pad=.12",facecolor=title_faces[i],edgecolor="none"))
        ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks(yticks); ax.grid(False)
        ax.spines[["top","right"]].set_visible(False); axes.append(ax)
    handles=[Patch(facecolor=METHOD_COLORS[m],edgecolor="#7A35B5" if m=="SpaCRAFT" else "none",linewidth=1,label=m) for m in METRIC_METHODS]
    legend_ax=fig.add_subplot(outer[3]); legend_ax.axis("off")
    legend=legend_ax.legend(handles=handles,loc="center",ncol=7,frameon=False,fontsize=8,
                            columnspacing=.8,handlelength=1.05,handletextpad=.4)
    return [group_ax]+axes+[legend_ax],legend


def draw_panel_b(fig,spec,c,cell_order,cell_pal,labels_by_method):
    # Latest manuscript layout: ground truth spans the full width; method
    # panels form a 4 x 2 block, with six slices arranged 2 x 3 per method.
    outer=spec.subgridspec(2,1,height_ratios=[.29,.71],hspace=.055)
    axes=[]
    method_colors={m:matched_cluster_colors(labels_by_method[m],c,cell_order,cell_pal)[0] for m in SPATIAL_METHODS}
    gt=outer[0].subgridspec(2,1,height_ratios=[.20,1],hspace=0)
    gt_title=fig.add_subplot(gt[0]); gt_title.axis("off")
    gt_title.text(.5,.40,"Ground truth",ha="center",va="center",fontsize=9,fontweight="bold")
    gt_grid=gt[1].subgridspec(1,6,wspace=.045)
    for col,sid in enumerate(SLICES):
        ax=fig.add_subplot(gt_grid[0,col]); mask=(c.slice_id==sid).to_numpy(); q=c.loc[mask]
        spatial(ax,q,q.cell_type.map(cell_pal).to_numpy(),size=1.65); axes.append(ax)
        ax.set_title(sid,pad=2.0,fontsize=7.4)

    method_grid=outer[1].subgridspec(2,4,wspace=.055,hspace=.10)
    legend=None
    for index,method in enumerate(SPATIAL_METHODS):
        cell=method_grid[index//4,index%4].subgridspec(3,3,height_ratios=[.16,1,1],wspace=.015,hspace=.035)
        title_ax=fig.add_subplot(cell[0,:]); title_ax.axis("off")
        title_ax.text(.5,.48,method,ha="center",va="center",fontsize=8.2)
        for pos,sid in enumerate(SLICES):
            ax=fig.add_subplot(cell[1+pos//3,pos%3]); mask=(c.slice_id==sid).to_numpy(); q=c.loc[mask]
            spatial(ax,q,method_colors[method][mask],size=1.55); axes.append(ax)
    legend_ax=fig.add_subplot(method_grid[1,3]); legend_ax.axis("off")
    legend=legend_ax.legend(handles=legend_handles(cell_pal,cell_order),loc="center",ncol=2,frameon=False,
                            fontsize=7.3,columnspacing=.65,handletextpad=.30,labelspacing=.20,
                            title="Cell type / matched cluster",title_fontsize=8.0)
    return [gt_title]+axes+[legend_ax],legend


def compute_umaps(c,embeddings):
    out={}
    for method in UMAP_METHODS:
        xy=umap.UMAP(**UMAP_CONFIG).fit_transform(embeddings[method])
        if xy.shape!=(len(c),2) or not np.isfinite(xy).all(): raise ValueError(f"{method}: invalid UMAP")
        out[method]=xy
        print(f"UMAP_COMPLETE={method}",flush=True)
    return out


def add_umap_axes(ax):
    # Exactly one coordinate indicator per row: only the leftmost SpaCRAFT map.
    x0,y0=.08,.10
    ax.annotate("",xy=(x0+.18,y0),xytext=(x0,y0),xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>",lw=.75,color="#333",shrinkA=0,shrinkB=0))
    ax.annotate("",xy=(x0,y0+.18),xytext=(x0,y0),xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>",lw=.75,color="#333",shrinkA=0,shrinkB=0))
    ax.text(x0+.19,y0-.005,"UMAP1",transform=ax.transAxes,fontsize=5.5,ha="left",va="top")
    ax.text(x0-.012,y0+.19,"UMAP2",transform=ax.transAxes,fontsize=5.5,ha="right",va="bottom",rotation=90)


def draw_panel_c(fig,spec,c,cell_order,cell_pal,umaps):
    outer=spec.subgridspec(4,1,height_ratios=[1,1,.11,.20],hspace=.075)
    grid=outer[:2].subgridspec(2,7,wspace=.045,hspace=.12); axes=[]; top_axes=[]
    slice_colors=c.slice_id.map(SLICE_COLORS).to_numpy(); cell_colors=c.cell_type.map(cell_pal).to_numpy()
    for col,method in enumerate(UMAP_METHODS):
        xy=umaps[method]
        for row,colors in enumerate((slice_colors,cell_colors)):
            ax=fig.add_subplot(grid[row,col]); ax.scatter(xy[:,0],xy[:,1],c=colors,s=.72,linewidths=0,rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if row==0: top_axes.append(ax)
            if col==0: ax.set_ylabel("Slice" if row==0 else "Cell type",fontweight="bold",rotation=90,labelpad=4)
            if col==0: add_umap_axes(ax)
            axes.append(ax)
    # Keep all method headings on one baseline even though equal-aspect UMAP
    # boxes have slightly different rendered heights.
    title_y=outer[0].get_position(fig).y1+.002
    for method,ax in zip(UMAP_METHODS,top_axes):
        box=ax.get_position()
        fig.text((box.x0+box.x1)/2,title_y,method,ha="center",va="bottom",fontsize=7.6)
    slice_ax=fig.add_subplot(outer[2]); slice_ax.axis("off")
    slice_ax.legend(handles=legend_handles(SLICE_COLORS,SLICES),loc="center",ncol=6,frameon=False,
                    fontsize=6.8,columnspacing=.6,handletextpad=.25,title="Slice",title_fontsize=7)
    cell_ax=fig.add_subplot(outer[3]); cell_ax.axis("off")
    cell_ax.legend(handles=legend_handles(cell_pal,cell_order),loc="center",ncol=6,frameon=False,
                   fontsize=6.5,columnspacing=.5,handletextpad=.22,title="Cell type",title_fontsize=7)
    return axes+[slice_ax,cell_ax]


def main():
    require_inputs(); c=canonical(); cell_order,cell_pal=cell_palette(c)
    embeddings={}; labels={}
    for method in METRIC_METHODS:
        embeddings[method],labels[method]=aligned_table(method,c)
    umaps=compute_umaps(c,embeddings)

    fig=plt.figure(figsize=(14.1,12.6),facecolor="white")
    master=fig.add_gridspec(3,1,height_ratios=[2.50,4.18,3.35],hspace=.050,
                            left=.055,right=.992,top=.992,bottom=.018)
    a_axes,a_leg=draw_panel_a(fig,master[0])
    b_axes,b_leg=draw_panel_b(fig,master[1],c,cell_order,cell_pal,labels)
    c_axes=draw_panel_c(fig,master[2],c,cell_order,cell_pal,umaps)
    fig.text(.012,master[0].get_position(fig).y1,"a",ha="left",va="top",fontsize=18,fontweight="bold")
    fig.text(.012,master[1].get_position(fig).y1,"b",ha="left",va="top",fontsize=18,fontweight="bold")
    fig.text(.012,master[2].get_position(fig).y1,"c",ha="left",va="top",fontsize=18,fontweight="bold")
    PNG.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(PNG,dpi=600,bbox_inches="tight",pad_inches=.04,facecolor="white")
    fig.savefig(PDF,bbox_inches="tight",pad_inches=.04,facecolor="white")
    plt.close(fig)
    print(f"PNG={PNG}\nPDF={PDF}")


if __name__=="__main__": main()

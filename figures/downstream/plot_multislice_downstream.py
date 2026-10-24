#!/usr/bin/env python3
"""Assemble the formal SpaCRAFT multi-slice downstream figure.

The script reads committed/precomputed plotting inputs only.  It does not run
analysis, select motifs, alter module membership, or train a model.
"""
import argparse
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
PROJECT_ROOT=args.project_root.resolve()
ROOT=PROJECT_ROOT/"results/downstream/multi_slice"
OUT=args.output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
OBS=PROJECT_ROOT/"results/multi_slice/canonical/six_slices_obs.csv"
FOUR_OBS=PROJECT_ROOT/"results/multi_slice/canonical/e11_e18_obs.csv"
LABELS=PROJECT_ROOT/"results/multi_slice/six_slices/SpaCRAFT/cluster_labels.csv"
CLUSTERPAL=ROOT/"cluster_color_palette.csv"
SCORES=ROOT/"C5_spot_module_scores.csv.gz"
MOTIFS=ROOT/"C5_module_motif_display_data.csv"
MOTIF_CACHE=ROOT/"panel_f_motif_spot_scores.csv.gz"
PROPORTIONS=ROOT/"cluster_proportions_by_slice.csv"
HEATMAP=ROOT/"C5_modules_row_zscore_six_slices.csv.gz"
MEMBERSHIP=ROOT/"C5_all_modules_membership.csv"
PROFILES=ROOT/"C5_module_profiles_by_stage_descriptive.csv"
ORDER=["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
STAGES=["E12.5","E12.5","E13.5","E13.5","E15.5","E15.5"]
STAGE_COLORS={"E12.5":"#4C78A8","E13.5":"#F2CF5B","E15.5":"#E45756"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.titlesize":8,
                     "pdf.fonttype":42})
obs=pd.read_csv(OBS)
lab=pd.read_csv(LABELS)[["canonical_index","cluster"]]
joined=obs.merge(lab,on="canonical_index",validate="one_to_one")
four_obs=pd.read_csv(FOUR_OBS)
cell_types=list(dict.fromkeys(list(obs.ground_truth_label)+list(four_obs.ground_truth_label)))
cm=plt.get_cmap("tab20",20);cm2=plt.get_cmap("tab20b",20)
cell_palette=[matplotlib.colors.to_hex(cm(i)) for i in range(20)]+[matplotlib.colors.to_hex(cm2(i)) for i in range(20)]
cell_df=pd.DataFrame({"cell_type":cell_types,"color":cell_palette[:len(cell_types)]})
cell_colors=dict(zip(cell_df.cell_type,cell_df.color))
if not CLUSTERPAL.exists():
    raise FileNotFoundError(CLUSTERPAL)
cluster_df=pd.read_csv(CLUSTERPAL)
cluster_colors={int(r.cluster):r.color_hex for r in cluster_df.itertuples()}
scores=pd.read_csv(SCORES)
motifs=pd.read_csv(MOTIFS)
assert joined.cluster.notna().all() and scores.canonical_index.is_unique
assert len(joined)==17787
assert int((joined.cluster==5).sum())==3208

def build_motif_spot_scores():
    """Compute accessibility of motif-bearing tested peaks relative to all tested peaks."""
    # Reuse the frozen motif-score cache; this assembly must not recompute
    # any analytical quantity.
    cache=MOTIF_CACHE
    if cache.exists():
        out=pd.read_csv(cache)
        assert {"SP8_motif_score","NFYC_motif_score"} <= set(out)
        return out
    raise FileNotFoundError(f"Frozen motif-score cache is required: {cache}")

motif_spot_scores=build_motif_spot_scores()
scores=scores.merge(motif_spot_scores,on="canonical_index",validate="one_to_one")

def clean(ax):
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)

def draw_b(fig,spec):
    prop=pd.read_csv(PROPORTIONS)
    grid=spec.subgridspec(5,1,height_ratios=[.12,.09,.78,.12,.20],hspace=.05)
    title_ax=fig.add_subplot(grid[0]);clean(title_ax)
    title_ax.text(.5,.55,"Cluster composition",ha="center",va="center",fontsize=8)
    stage_grid=grid[1].subgridspec(1,3,wspace=.02)
    stage_axes=[]
    for j,stage in enumerate(("E12.5","E13.5","E15.5")):
        stage_ax=fig.add_subplot(stage_grid[j]);clean(stage_ax)
        stage_ax.add_patch(Rectangle((0,.12),1,.22,transform=stage_ax.transAxes,
                                     color=STAGE_COLORS[stage],lw=0))
        stage_ax.text(.5,.52,stage,ha="center",va="bottom",fontsize=5.8)
        stage_axes.append(stage_ax)
    ax=fig.add_subplot(grid[2])
    bottom=np.zeros(len(ORDER),float)
    x=np.arange(len(ORDER))
    for cluster in sorted(cluster_colors):
        values=(prop[prop.cluster.eq(cluster)].set_index("slice")
                .reindex(ORDER).cluster_proportion_within_slice.fillna(0).to_numpy())
        ax.bar(x,values,bottom=bottom,width=.82,color=cluster_colors[cluster],
               edgecolor="white",linewidth=.15,label=f"C{cluster}")
        bottom+=values
    ax.set_xticks(x,ORDER,rotation=45,ha="right",fontsize=5.6)
    ax.set_ylim(0,1);ax.set_ylabel("Proportion",labelpad=2)
    ax.spines[["top","right"]].set_visible(False)
    spacer_ax=fig.add_subplot(grid[3]);clean(spacer_ax);spacer_ax.patch.set_visible(False)
    legend_ax=fig.add_subplot(grid[4]);clean(legend_ax)
    legend_ax.patch.set_visible(False)
    handles=[Rectangle((0,0),1,1,color=cluster_colors[k],label=f"C{k}") for k in sorted(cluster_colors)]
    legend_ax.legend(handles=handles,ncol=6,frameon=False,fontsize=6.2,loc="center",
                     columnspacing=.72,handletextpad=.28,labelspacing=.55,
                     handlelength=1.45,handleheight=1.05,borderaxespad=0)
    return [title_ax,*stage_axes,ax,spacer_ax,legend_ax]

def draw_c(fig,spec):
    vertical=spec.subgridspec(3,1,height_ratios=[.07,.78,.15],hspace=0)
    grid=vertical[1].subgridspec(1,2,width_ratios=[1,.07],wspace=.11)
    ax=fig.add_subplot(grid[0]);cax=fig.add_subplot(grid[1])
    heatmap=pd.read_csv(HEATMAP)
    membership=pd.read_csv(MEMBERSHIP).sort_values(["module","phase2_rank"],kind="mergesort")
    heatmap=heatmap.set_index("peak_id").loc[membership.peak_id].reset_index()
    image=ax.imshow(heatmap[ORDER].to_numpy(),aspect="auto",interpolation="nearest",
                    cmap="coolwarm",vmin=-2.5,vmax=2.5)
    ax.axhline(125.5,color="black",lw=.7)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(6)))
    ax.xaxis.set_major_formatter(FixedFormatter(ORDER))
    plt.setp(ax.get_xticklabels(),rotation=70,ha="right",va="top",rotation_mode="anchor")
    ax.tick_params(axis="x",labelsize=6.0,pad=2);ax.set_yticks([])
    ax.set_title("Ordered accessibility modules",pad=19,fontsize=8.4)
    ax.text(-.07,1-63/232,"Early-high\n(n=126)",transform=ax.transAxes,ha="right",va="center",fontsize=6.3)
    ax.text(-.07,53/232,"Late-high\n(n=106)",transform=ax.transAxes,ha="right",va="center",fontsize=6.3)
    for index,stage in enumerate(STAGES):
        ax.add_patch(Rectangle((index-.5,1.005),1,.038,transform=ax.get_xaxis_transform(),
                               color=STAGE_COLORS[stage],clip_on=False,lw=0))
    for center,stage in zip((.5,2.5,4.5),("E12.5","E13.5","E15.5")):
        ax.text(center,1.024,stage,transform=ax.get_xaxis_transform(),ha="center",va="center",fontsize=6.2,clip_on=False)
    fig.colorbar(image,cax=cax);cax.set_title("z-score",fontsize=6.5,pad=5);cax.tick_params(labelsize=6.0,pad=1)
    return [ax,cax]

def draw_d(fig,spec):
    vertical=spec.subgridspec(3,1,height_ratios=[.07,.82,.11],hspace=0)
    ax=fig.add_subplot(vertical[1]);profiles=pd.read_csv(PROFILES)
    colors={"Module_EarlyHigh":"#2166ac","Module_LateHigh":"#b2182b"}
    labels={"Module_EarlyHigh":"Early-high","Module_LateHigh":"Late-high"}
    for module in ("Module_EarlyHigh","Module_LateHigh"):
        q=profiles[profiles.module.eq(module)].copy();q["x"]=q.stage.map({"E12.5":0,"E13.5":1,"E15.5":2});q=q.sort_values("x")
        ax.plot(q.x,q.stage_mean_row_zscore,color=colors[module],marker="o",ms=4.2,lw=1.4,label=labels[module])
    ax.axhline(0,color=".75",lw=.7);ax.set_xticks([0,1,2],["E12.5","E13.5","E15.5"])
    ax.set_xlabel("Stage",labelpad=2);ax.set_ylabel("Module score",labelpad=2)
    ax.set_title("Module trajectories",pad=8);ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,.03),fontsize=5.5)
    ax.spines[["top","right"]].set_visible(False)
    return [ax]

def draw_e(fig,spec):
    outer=spec.subgridspec(3,1,height_ratios=[.03,1,.04],hspace=0)
    grid=outer[1].subgridspec(1,2,width_ratios=[1,.07],wspace=.13)
    ax=fig.add_subplot(grid[0]);cax=fig.add_subplot(grid[1])
    display=motifs.copy();names=display.TF_symbol.tolist();matrix=display[["EarlyHigh_log2OR","LateHigh_log2OR"]].to_numpy()
    maximum=max(abs(matrix.min()),abs(matrix.max()))
    im=ax.imshow(matrix,aspect="auto",cmap="coolwarm",
                 norm=TwoSlopeNorm(vcenter=0,vmin=-maximum,vmax=maximum),interpolation="nearest")
    ax.set_xticks([0,1],["Early-high","Late-high"],rotation=0)
    ax.set_yticks(np.arange(len(names)),names);ax.tick_params(axis="y",pad=2)
    ax.set_title("TF motif enrichment",pad=7)
    for i,row in display.reset_index(drop=True).iterrows():
        for j,p in enumerate((row.EarlyHigh_FDR,row.LateHigh_FDR)):
            if p<=.05:ax.scatter(j,i,s=8,c="black")
    for boundary in (7.5,9.5):ax.axhline(boundary,color="white",lw=.45)
    cb=fig.colorbar(im,cax=cax);cb.set_label("log2 odds ratio",fontsize=7.0,labelpad=2)
    cax.tick_params(labelsize=6.5,pad=1)
    return [ax,cax]

def style_spatial(ax):
    ax.set_aspect("equal",adjustable="datalim");ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_linewidth(.25);s.set_color("#cccccc")

def draw_a(fig,spec):
    # Keep the two spatial rows dominant, but reserve independent corridors for
    # the stage strip, cluster legend, and ground-truth legend.  The spatial
    # axes lose only a small amount of their pale frame/whitespace footprint.
    gs=spec.subgridspec(5,6,height_ratios=[.16,1,1,.095,.225],hspace=.18,wspace=.045)
    gt_axes=[];cl_axes=[]
    for j,sec in enumerate(ORDER):
        sub=joined[joined.slice_id.eq(sec)]
        ag=fig.add_subplot(gs[1,j]);ac=fig.add_subplot(gs[2,j]);gt_axes.append(ag);cl_axes.append(ac)
        for ct in cell_df.cell_type:
            q=sub[sub.ground_truth_label.eq(ct)]
            if len(q):ag.scatter(q.spatial_x,q.spatial_y,s=.48,c=cell_colors[ct],edgecolors="none",rasterized=True)
        for k in sorted(cluster_colors):
            q=sub[sub.cluster.eq(k)]
            if len(q):ac.scatter(q.spatial_x,q.spatial_y,s=.48,c=cluster_colors[k],edgecolors="none",rasterized=True)
        for ax in (ag,ac):style_spatial(ax)
        # Slice names remain below the stage strip with an explicit title pad.
        ag.set_title(sec,fontsize=5.2,pad=2.0)
    for start,stage in zip((0,2,4),("E12.5","E13.5","E15.5")):
        ax=fig.add_subplot(gs[0,start:start+2]);clean(ax)
        # Titles from the spatial row may extend into the inter-row corridor;
        # a transparent stage Axes prevents its background from masking them.
        ax.patch.set_visible(False)
        ax.add_patch(Rectangle((0,.34),1,.16,transform=ax.transAxes,color=STAGE_COLORS[stage],lw=0))
        ax.text(.5,.68,stage,transform=ax.transAxes,ha="center",va="bottom",fontsize=6)
    gt_axes[0].text(-.25,.5,"Ground truth",rotation=90,transform=gt_axes[0].transAxes,ha="center",va="center",fontsize=6.2)
    cl_axes[0].text(-.25,.5,"SpaCRAFT clusters",rotation=90,transform=cl_axes[0].transAxes,ha="center",va="center",fontsize=6.2)
    cluster_la=fig.add_subplot(gs[3,:]);clean(cluster_la)
    ch=[Line2D([0],[0],marker="o",ls="",ms=2.8,mfc=cluster_colors[k],mec="none",label=f"C{k}") for k in sorted(cluster_colors)]
    leg1=cluster_la.legend(handles=ch,ncol=11,loc="center",frameon=False,fontsize=4.8,
                           columnspacing=.45,handletextpad=.15,title="SpaCRAFT clusters",
                           title_fontsize=5.3,borderaxespad=0)
    ground_la=fig.add_subplot(gs[4,:]);clean(ground_la)
    present=[x for x in cell_df.cell_type if x in set(joined.ground_truth_label)]
    gh=[Line2D([0],[0],marker="o",ls="",ms=2.5,mfc=cell_colors[x],mec="none",label=x) for x in present]
    leg2=ground_la.legend(handles=gh,ncol=6,loc="center",frameon=False,fontsize=4.1,
                          columnspacing=.4,handletextpad=.15,title="Ground-truth cell types",
                          title_fontsize=4.8,borderaxespad=0)
    return gt_axes+cl_axes+[cluster_la,ground_la], {
        "cluster_legend": leg1,
        "ground_legend": leg2,
        "gt_axes": gt_axes,
    }

def draw_f(fig,spec):
    # The main spatial programs stay in the right-hand body.
    outer=spec.subgridspec(5,1,height_ratios=[.15,.20,1,1,1],hspace=.13)
    title_ax=fig.add_subplot(outer[0]);clean(title_ax)
    title_ax.text(.5,.62,"Spatial programs",
                  ha="center",va="center",fontsize=8.5)
    axes=[title_ax]

    # Spatial-program header: all six slices. Colorbars use a deliberately
    # narrow final column for a consistent slim style.
    header=outer[1].subgridspec(1,8,width_ratios=[.23,1,1,1,1,1,1,.035],wspace=.075)
    hlabel=fig.add_subplot(header[0]);clean(hlabel);axes.append(hlabel)
    for j,(stage,sec) in enumerate(zip(STAGES,ORDER),1):
        ha=fig.add_subplot(header[j]);clean(ha)
        ha.text(.5,.52,f"{stage}\n{sec}",ha="center",va="center",fontsize=4.7,
                linespacing=1.05)
        axes.append(ha)
    hblank=fig.add_subplot(header[7]);clean(hblank);axes.append(hblank)

    spatial_rows=[
        ("C5 spatial\nclustering",None),
        ("Early-high\nmodule score","EarlyHigh_module_score"),
        ("Late-high\nmodule score","LateHigh_module_score"),
    ]
    for row_index,(label,valcol) in enumerate(spatial_rows,2):
        gs=outer[row_index].subgridspec(
            1,8,width_ratios=[.23,1,1,1,1,1,1,.035],wspace=.075
        )
        la=fig.add_subplot(gs[0]);clean(la)
        la.text(.52,.5,label,rotation=90,ha="center",va="center",fontsize=5.0,
                linespacing=1.0)
        axes.append(la)
        if valcol:
            foreground=scores.loc[scores.is_C5.astype(bool),valcol].to_numpy(float)
            vmin,vmax=np.nanquantile(foreground,[.02,.98])
        last=None
        for j,sec in enumerate(ORDER,1):
            ax=fig.add_subplot(gs[j]);q=scores[scores.slice.eq(sec)];inc=q.is_C5.astype(bool)
            ax.scatter(q.loc[~inc,"spatial_x"],q.loc[~inc,"spatial_y"],s=.32,
                       c="#bdbdbd",alpha=.92,edgecolors="none",rasterized=True)
            if valcol:
                last=ax.scatter(q.loc[inc,"spatial_x"],q.loc[inc,"spatial_y"],s=.84,
                                c=q.loc[inc,valcol],cmap="viridis",vmin=vmin,vmax=vmax,
                                alpha=1.0,edgecolors="none",rasterized=True)
            else:
                last=ax.scatter(q.loc[inc,"spatial_x"],q.loc[inc,"spatial_y"],s=.88,
                                c=cluster_colors[5],alpha=1.0,edgecolors="none",rasterized=True)
            style_spatial(ax);axes.append(ax)
        ca=fig.add_subplot(gs[7]);axes.append(ca)
        if valcol:
            cb=fig.colorbar(last,cax=ca);ca.tick_params(labelsize=4.2,pad=1,length=2)
            cb.set_ticks([vmin,(vmin+vmax)/2,vmax])
        else:clean(ca)

    return axes

def draw_g(fig,motif_spec):
    """Draw the unchanged representative-motif maps as an independent panel."""
    outer=motif_spec.subgridspec(2,1,height_ratios=[.13,1],hspace=.06)
    title_ax=fig.add_subplot(outer[0]);clean(title_ax)
    title_ax.text(.5,.52,"Representative motifs",ha="center",va="center",fontsize=8.5)
    axes=[title_ax]
    # The first three maps are SP8 and the next three are NFYC, each ordered
    # E12.5, E13.5, E15.5. Scores, ordering, ranges and colour logic are frozen.
    representative_slices=[ORDER[0],ORDER[2],ORDER[4]]
    motif_defs=[("SP8","Early-high","SP8_motif_score"),
                ("NFYC","Late-high","NFYC_motif_score")]
    map_defs=[]
    for tf,direction,col in motif_defs:
        for sec in representative_slices:
            map_defs.append((tf,direction,col,sec))
    # Dedicated narrow spacer columns on the outer side of each colorbar keep
    # full tick strings away from adjacent maps without shrinking the maps.
    motif_grid=outer[1].subgridspec(
        1,13,
        width_ratios=[.25,1,1,1,.05,.025,.15,1,1,1,.05,.025,.16],
        wspace=.040,
    )
    row_label=fig.add_subplot(motif_grid[0]);clean(row_label)
    row_label.text(.55,.5,"Motif spatial",rotation=90,ha="center",va="center",
                   fontsize=5.2)
    axes.append(row_label)
    motif_limits={}
    for tf,_,col in motif_defs:
        values=scores.loc[scores.is_C5.astype(bool),col].to_numpy(float)
        motif_limits[tf]=tuple(np.nanquantile(values,[.02,.98]))
    map_columns=[1,2,3,7,8,9]
    motif_mappables={}
    for map_column,(tf,direction,col,sec) in zip(map_columns,map_defs):
        vmin,vmax=motif_limits[tf]
        ax=fig.add_subplot(motif_grid[map_column]);q=scores[scores.slice.eq(sec)];inc=q.is_C5.astype(bool)
        ax.scatter(q.loc[~inc,"spatial_x"],q.loc[~inc,"spatial_y"],s=.32,
                   c="#bdbdbd",alpha=.92,edgecolors="none",rasterized=True)
        motif_mappables[tf]=ax.scatter(
            q.loc[inc,"spatial_x"],q.loc[inc,"spatial_y"],s=.90,c=q.loc[inc,col],
            cmap="viridis",vmin=vmin,vmax=vmax,alpha=1.0,
            edgecolors="none",rasterized=True
        )
        style_spatial(ax)
        ax.set_title(f"{tf} · {direction}\n{STAGES[ORDER.index(sec)]}",
                     fontsize=4.65,pad=2.0,linespacing=1.05)
        axes.append(ax)
    for column,(tf,_,_) in zip((5,11),motif_defs):
        ca=fig.add_subplot(motif_grid[column]);cb=fig.colorbar(motif_mappables[tf],cax=ca)
        position=ca.get_position()
        ca.set_position([position.x0,position.y0+.10*position.height,
                         position.width,.80*position.height])
        vmin,vmax=motif_limits[tf]
        ca.set_title(tf,fontsize=4.2,pad=2)
        ca.yaxis.set_ticks_position("right")
        ca.tick_params(labelsize=3.8,pad=1.2,length=1.8,labelleft=False,labelright=True)
        ticks=[vmin,(vmin+vmax)/2,vmax]
        cb.set_ticks(ticks)
        ca.set_yticks(ticks)
        ca.set_yticklabels([f"{value:.5f}" for value in ticks],fontsize=3.8)
        for tick_label in ca.get_yticklabels():tick_label.set_clip_on(False)
        axes.append(ca)
    for column in (4,6,10,12):
        spacer=fig.add_subplot(motif_grid[column]);clean(spacer)
        spacer.patch.set_visible(False)
        axes.append(spacer)
    return axes

def add_letter(fig,axes,letter,dx=.008,dy=.006):
    box=matplotlib.transforms.Bbox.union([a.get_position() for a in axes])
    fig.text(box.x0-dx,box.y1+dy,letter,fontsize=12,fontweight="bold",ha="right",va="bottom")

fig=plt.figure(figsize=(15.5,10.35),facecolor="white")
master=fig.add_gridspec(3,1,height_ratios=[.92,.72,.31],hspace=.115,left=.022,right=.985,top=.965,bottom=.035)
top=master[0].subgridspec(1,4,width_ratios=[2.75,1.05,.82,1.00],wspace=.26)
middle=master[1].subgridspec(1,2,width_ratios=[.58,3.82],wspace=.18)
panels={}
panels['a'],panel_a_meta=draw_a(fig,top[0])
panels['b']=draw_b(fig,top[1])
panels['c']=draw_c(fig,top[2])
panels['d']=draw_d(fig,top[3])
panels['e']=draw_e(fig,middle[0,0])
panels['f']=draw_f(fig,middle[0,1])
panels['g']=draw_g(fig,master[2])
# Shift the complete panel-e artwork right while keeping its panel letter fixed.
e_box_before_shift=matplotlib.transforms.Bbox.union(
    [axis.get_position() for axis in panels['e']])
for axis in panels['e']:
    position=axis.get_position()
    axis.set_position([position.x0+.012,position.y0,position.width,position.height])
e_box_after_shift=matplotlib.transforms.Bbox.union(
    [axis.get_position() for axis in panels['e']])
for letter in 'acd':add_letter(fig,panels[letter],letter)
fig.text(e_box_before_shift.x0-.008,e_box_after_shift.y1+.006,"e",fontsize=12,
         fontweight="bold",ha="right",va="bottom")
# Keep the top-row labels b/c/d on one shared horizontal baseline.
b_box=matplotlib.transforms.Bbox.union([axis.get_position() for axis in panels['b']])
c_box=matplotlib.transforms.Bbox.union([axis.get_position() for axis in panels['c']])
fig.text(b_box.x0-.008,c_box.y1+.006,"b",fontsize=12,fontweight="bold",
         ha="right",va="bottom")
add_letter(fig,[panels['f'][0]],'f')
add_letter(fig,panels['g'],'g',dx=.006,dy=.005)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
cluster_box=panel_a_meta["cluster_legend"].get_window_extent(renderer)
ground_box=panel_a_meta["ground_legend"].get_window_extent(renderer)
legend_gap_px=cluster_box.y0-ground_box.y1
if legend_gap_px <= 0:
    raise RuntimeError(f"Panel a legend overlap remains: gap={legend_gap_px:.3f}px")

base=OUT/"Figure_multislice_downstream"
fig.savefig(base.with_suffix('.png'),dpi=300,facecolor='white')
fig.savefig(base.with_suffix('.pdf'),facecolor='white')
plt.close(fig)
for path in (base.with_suffix('.png'),base.with_suffix('.pdf')):
    assert path.exists() and path.stat().st_size>0
print(f"PNG={base.with_suffix('.png')}")
print(f"PDF={base.with_suffix('.pdf')}")

#!/usr/bin/env python3
"""Phase 1 frozen-cluster composition and coverage analysis (descriptive only)."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd

ROOT = MULTI_DOWNSTREAM_ROOT
OUT = ROOT / "results/composition_v1"
SIX = OUT / "six_slice"
FOUR = OUT / "four_slice"
COMB = OUT / "combined_audit"
META = OUT / "metadata"
for d in (SIX, FOUR, COMB, META):
    d.mkdir(parents=True, exist_ok=True)

SIX_OBS = MULTISLICE_RESULTS_ROOT / "canonical/six_slices_obs.csv"
SIX_LAB = MULTISLICE_RESULTS_ROOT / "six_slices/SpaCRAFT/cluster_labels.csv"
FOUR_OBS = MULTISLICE_RESULTS_ROOT / "canonical/e11_e18_obs.csv"
FOUR_LAB = MULTISLICE_RESULTS_ROOT / "e11_e18/SpaCRAFT/cluster_labels.csv"
SIX_ORDER = ["220403_C2","220403_D2","220403_A2","220403_B2","220327_A1","220327_A2"]
SIX_STAGE = {"220403_C2":"E12.5","220403_D2":"E12.5","220403_A2":"E13.5",
             "220403_B2":"E13.5","220327_A1":"E15.5","220327_A2":"E15.5"}
FOUR_ORDER = ["E11_S1","E13_S1","E15_S1","E18_S1"]

def load_join(obs_path, label_path, expected_n, k, order, stage_map=None):
    obs = pd.read_csv(obs_path)
    lab = pd.read_csv(label_path)
    required_obs = {"obs_name","slice_id","ground_truth_label","canonical_index"}
    assert required_obs <= set(obs.columns)
    assert {"obs_name","canonical_index","cluster"} <= set(lab.columns)
    assert len(obs) == len(lab) == expected_n
    assert not obs["canonical_index"].duplicated().any()
    assert not lab["canonical_index"].duplicated().any()
    assert set(obs["canonical_index"]) == set(range(expected_n))
    assert set(lab["canonical_index"]) == set(range(expected_n))
    joined = obs.merge(lab, on="canonical_index", suffixes=("_canonical","_label"),
                       validate="one_to_one")
    assert (joined["obs_name_canonical"] == joined["obs_name_label"]).all()
    joined = joined.rename(columns={"obs_name_canonical":"obs_name"})
    joined = joined.drop(columns="obs_name_label")
    assert sorted(joined["cluster"].unique().tolist()) == list(range(k))
    assert joined["slice_id"].notna().all() and joined["cluster"].notna().all()
    assert set(joined["slice_id"]) == set(order)
    joined["stage"] = (joined["slice_id"].map(stage_map) if stage_map
                       else joined["slice_id"].str.extract(r"^(E\d+)")[0])
    assert joined["stage"].notna().all()
    return joined.sort_values("canonical_index").reset_index(drop=True)

def coverage_level(n):
    return "ABSENT" if n == 0 else "VERY_LOW" if n < 30 else "LIMITED" if n < 100 else "ADEQUATE"

def normalized_entropy(x, k):
    p = x.value_counts(normalize=True).to_numpy()
    return float(-(p*np.log(p)).sum()/np.log(k)) if k > 1 else 0.0

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])

def js_distance(a, b):
    m = (a+b)/2
    def kl(x, y):
        keep=x>0
        return float(np.sum(x[keep]*np.log2(x[keep]/y[keep])))
    return float(np.sqrt((kl(a,m)+kl(b,m))/2))

def annotations(df, k, order):
    overall, by_slice = [], []
    for c in range(k):
        g = df[df.cluster.eq(c)]
        vc = g.ground_truth_label.value_counts()
        top = vc.head(3)
        row = {"cluster":c,"spot_count":len(g),"majority_cell_type":vc.index[0],
               "majority_fraction":vc.iloc[0]/len(g),"cell_type_purity":vc.iloc[0]/len(g),
               "normalized_entropy":normalized_entropy(g.ground_truth_label, vc.size)}
        for i in range(3):
            row[f"top{i+1}_cell_type"] = top.index[i] if i < len(top) else ""
            row[f"top{i+1}_proportion"] = top.iloc[i]/len(g) if i < len(top) else 0.0
        overall.append(row)
        for s in order:
            gs = g[g.slice_id.eq(s)]
            vcs = gs.ground_truth_label.value_counts()
            r = {"slice":s,"stage":gs.stage.iloc[0] if len(gs) else
                 (SIX_STAGE.get(s) or s.split("_")[0]),"cluster":c,"spot_count":len(gs)}
            if len(gs):
                r.update({"majority_cell_type":vcs.index[0],"majority_fraction":vcs.iloc[0]/len(gs),
                          "cell_type_purity":vcs.iloc[0]/len(gs),
                          "normalized_entropy":normalized_entropy(gs.ground_truth_label,vcs.size)})
                for i in range(3):
                    r[f"top{i+1}_cell_type"] = vcs.index[i] if i < len(vcs) else ""
                    r[f"top{i+1}_proportion"] = vcs.iloc[i]/len(gs) if i < len(vcs) else 0.0
            else:
                r.update({"majority_cell_type":"","majority_fraction":np.nan,
                          "cell_type_purity":np.nan,"normalized_entropy":np.nan})
                for i in range(3):
                    r[f"top{i+1}_cell_type"] = ""; r[f"top{i+1}_proportion"] = 0.0
            by_slice.append(r)
    return pd.DataFrame(overall), pd.DataFrame(by_slice)

def per_slice(df, k, order):
    idx = pd.MultiIndex.from_product([order, range(k)], names=["slice","cluster"])
    counts = (df.groupby(["slice_id","cluster"]).size().rename("spot_count")
              .reindex(idx, fill_value=0).reset_index())
    stage_map = df.drop_duplicates("slice_id").set_index("slice_id")["stage"]
    totals = df.groupby("slice_id").size()
    counts.insert(1,"stage",counts["slice"].map(stage_map))
    counts["slice_total_spots"] = counts["slice"].map(totals)
    props = counts.copy()
    props["cluster_proportion_within_slice"] = props.spot_count/props.slice_total_spots
    return counts, props

six = load_join(SIX_OBS,SIX_LAB,17787,11,SIX_ORDER,SIX_STAGE)
four = load_join(FOUR_OBS,FOUR_LAB,7118,16,FOUR_ORDER)

six_counts, six_props = per_slice(six,11,SIX_ORDER)
four_counts, four_props = per_slice(four,16,FOUR_ORDER)
six_counts.to_csv(SIX/"cluster_counts_by_slice.csv",index=False)
six_props.to_csv(SIX/"cluster_proportions_by_slice.csv",index=False)
four_counts.to_csv(FOUR/"cluster_counts_by_slice.csv",index=False)
four_props.to_csv(FOUR/"cluster_proportions_by_slice.csv",index=False)

stage_rows = []
for stage in ["E12.5","E13.5","E15.5"]:
    ss = [s for s in SIX_ORDER if SIX_STAGE[s] == stage]
    for c in range(11):
        v = six_props.query("slice in @ss and cluster == @c").set_index("slice")[
            "cluster_proportion_within_slice"]
        stage_rows.append({"stage":stage,"cluster":c,"slice_1":ss[0],
            "slice_1_proportion":v[ss[0]],"slice_2":ss[1],"slice_2_proportion":v[ss[1]],
            "stage_mean":v.mean(),"stage_min":v.min(),"stage_max":v.max(),
            "stage_sd":v.std(ddof=1),"n_slices":2})
stage_desc = pd.DataFrame(stage_rows)
stage_desc.to_csv(SIX/"cluster_proportions_by_stage_descriptive.csv",index=False)

consistency, differences = [], []
for stage in ["E12.5","E13.5","E15.5"]:
    ss = [s for s in SIX_ORDER if SIX_STAGE[s] == stage]
    a = six_props.query("slice == @ss[0]").sort_values("cluster").cluster_proportion_within_slice.to_numpy()
    b = six_props.query("slice == @ss[1]").sort_values("cluster").cluster_proportion_within_slice.to_numpy()
    pear = corr(a,b)
    spear = corr(pd.Series(a).rank().to_numpy(),pd.Series(b).rank().to_numpy())
    js = js_distance(a,b)
    md = np.abs(a-b).max()
    status = ("HIGH_DESCRIPTIVE_CONSISTENCY" if pear>=.9 and js<=.1 and md<=.1 else
              "MODERATE_DESCRIPTIVE_CONSISTENCY" if pear>=.7 and js<=.2 and md<=.2 else
              "LOW_DESCRIPTIVE_CONSISTENCY")
    ra = pd.Series(a).rank(ascending=False,method="average").to_numpy()
    rb = pd.Series(b).rank(ascending=False,method="average").to_numpy()
    consistency.append({"stage":stage,"slice_1":ss[0],"slice_2":ss[1],
        "pearson_correlation":pear,"spearman_correlation":spear,
        "l1_distance":np.abs(a-b).sum(),"jensen_shannon_distance":js,
        "max_absolute_proportion_difference":md,
        "exact_rank_agreement_fraction":np.mean(ra==rb),
        "top3_cluster_overlap_count":len(set(np.argsort(-a)[:3])&set(np.argsort(-b)[:3])),
        "consistency_status":status,
        "interpretation":"descriptive only; not evidence of biological replication"})
    for c in range(11):
        differences.append({"stage":stage,"slice_1":ss[0],"slice_2":ss[1],"cluster":c,
            "slice_1_proportion":a[c],"slice_2_proportion":b[c],
            "absolute_proportion_difference":abs(a[c]-b[c]),
            "direction_slice2_minus_slice1":b[c]-a[c],"rank_slice_1":ra[c],"rank_slice_2":rb[c]})
pd.DataFrame(consistency).to_csv(SIX/"same_stage_slice_consistency.csv",index=False)
pd.DataFrame(differences).to_csv(SIX/"same_stage_cluster_difference.csv",index=False)

def coverage_outputs(counts, out, stages, k):
    cs = counts.copy()
    cs["coverage_level"] = cs.spot_count.map(coverage_level)
    cstage = (counts.groupby(["stage","cluster"],as_index=False).spot_count.sum())
    stage_idx = pd.MultiIndex.from_product([stages,range(k)],names=["stage","cluster"])
    cstage = cstage.set_index(["stage","cluster"]).reindex(stage_idx,fill_value=0).reset_index()
    cstage["coverage_level"] = cstage.spot_count.map(coverage_level)
    cs.to_csv(out/"cluster_slice_coverage.csv",index=False)
    cstage.to_csv(out/"cluster_stage_coverage.csv",index=False)
    elig = []
    for c in range(k):
        slv = cs[cs.cluster.eq(c)].coverage_level.tolist()
        stv = cstage[cstage.cluster.eq(c)].coverage_level.tolist()
        peak = all(x=="ADEQUATE" for x in slv) and all(x=="ADEQUATE" for x in stv)
        gene = all(x in ("ADEQUATE","LIMITED") for x in slv) and all(x=="ADEQUATE" for x in stv)
        reasons=[]
        if not peak: reasons.append("not >=100 spots in every slice/stage")
        if not gene: reasons.append("not >=30 in every slice or not >=100 in every stage")
        elig.append({"cluster":c,"stage_associated_peak_analysis_eligible":peak,
                     "gene_activity_dynamics_eligible":gene,
                     "motif_GO_analysis_eligible":peak,
                     "minimum_slice_spots":int(cs[cs.cluster.eq(c)].spot_count.min()),
                     "minimum_stage_spots":int(cstage[cstage.cluster.eq(c)].spot_count.min()),
                     "limitation_reason":"; ".join(reasons) if reasons else "none"})
    pd.DataFrame(elig).to_csv(out/"cluster_downstream_eligibility.csv",index=False)
    return cs,cstage,pd.DataFrame(elig)

six_cov,six_stage_cov,six_elig = coverage_outputs(six_counts,SIX,["E12.5","E13.5","E15.5"],11)
four_cov,four_stage_cov,four_elig = coverage_outputs(four_counts,FOUR,["E11","E13","E15","E18"],16)

six_ann,six_ann_sl = annotations(six,11,SIX_ORDER)
four_ann,four_ann_sl = annotations(four,16,FOUR_ORDER)
six_ann.to_csv(SIX/"cluster_celltype_annotation_overall.csv",index=False)
six_ann_sl.to_csv(SIX/"cluster_celltype_annotation_by_slice.csv",index=False)
four_ann.to_csv(FOUR/"cluster_celltype_annotation_overall.csv",index=False)
four_ann_sl.to_csv(FOUR/"cluster_celltype_annotation_by_slice.csv",index=False)

mapping=[]
for _,a in six_ann.iterrows():
    candidates=four_ann[four_ann.majority_cell_type.eq(a.majority_cell_type)]
    if len(candidates)==1 and a.cell_type_purity>=.7 and candidates.iloc[0].cell_type_purity>=.7:
        b=candidates.iloc[0]; status="ANNOTATION_SUPPORTED_CANDIDATE"
        four_id=int(b.cluster)
    else:
        four_id=""; status="NO_RELIABLE_CLUSTER_ID_MAPPING"
    mapping.append({"six_slice_cluster":int(a.cluster),"six_majority_cell_type":a.majority_cell_type,
        "six_purity":a.cell_type_purity,"four_slice_cluster_candidate":four_id,
        "mapping_status":status,"note":"cluster IDs are never treated as equivalent"})
pd.DataFrame(mapping).to_csv(COMB/"cross_dataset_annotation_mapping.csv",index=False)

summary={"analysis_type":"DESCRIPTIVE_COMPOSITION_ANALYSIS","six_joined":len(six),
         "four_joined":len(four),"six_k":11,"four_k":16,
         "six_proportion_max_error":float((six_props.groupby("slice").cluster_proportion_within_slice.sum()-1).abs().max()),
         "four_proportion_max_error":float((four_props.groupby("slice").cluster_proportion_within_slice.sum()-1).abs().max()),
         "six_same_stage_statuses":pd.DataFrame(consistency).consistency_status.tolist(),
         "six_peak_eligible_clusters":six_elig.query("stage_associated_peak_analysis_eligible").cluster.tolist(),
         "six_gene_eligible_clusters":six_elig.query("gene_activity_dynamics_eligible").cluster.tolist(),
         "cross_dataset_mapping_status":"NO_RELIABLE_CLUSTER_ID_MAPPING",
         "formal_inference_performed":False,"spot_as_replicate_used":False}
(META/"analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

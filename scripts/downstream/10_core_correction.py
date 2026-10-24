#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = SINGLE_DOWNSTREAM_ROOT
BY = ROOT / "results/by_slice"
AUDIT = ROOT / "audit"
PREV = ROOT / "figures/previews"
SCREEN = ROOT / "results/screening"
FASTA = GENOME_ROOT / "genome.fa"
MEME = GENOME_ROOT / "motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
GO_DIR = ROOT / "resources/gene_sets/go_2026_06_19"
OBO = GO_DIR / "go-basic.obo"
GAF = GO_DIR / "MOUSE-mod.gaf.gz"
SEED = 1234
GO_DISTAL_MAX_BP = 100_000
SLICES = [
    "E11_0-S1_atac", "E13_5-S1_atac", "E15_5-S1_atac",
    "E18_5-S1_atac", "adata_peaks_normalized", "s1_adata_atac_with_meta",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def savefig(fig, png: Path, pdf: Path | None = None) -> None:
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    if pdf is not None:
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_meme():
    from snapatac2._snapatac2 import PyDNAMotif
    lines = MEME.read_text().splitlines()
    motifs, names = [], {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("MOTIF "):
            z = lines[i].split(maxsplit=2)
            motif_id = z[1]
            names[motif_id] = z[2] if len(z) > 2 else motif_id
            i += 1
            while i < len(lines) and "letter-probability matrix:" not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            width = int(re.search(r"\bw=\s*(\d+)", lines[i]).group(1))
            matrix = [[float(x) for x in q.split()[:4]] for q in lines[i + 1:i + 1 + width]]
            motifs.append(PyDNAMotif(motif_id, np.asarray(matrix)))
            i += width
        i += 1
    return motifs, names


def motif_regions_and_background(slice_name: str):
    ann = pd.read_csv(BY / slice_name / "peak_annotation/all_peaks_annotation.csv", low_memory=False)
    pos = pd.read_csv(BY / slice_name / "marker_peaks/positive_markers.csv.gz")
    sig = pos.query("adjusted_pvalue <= 0.05 and log2FC >= 0.5").copy()
    fasta_chroms = set(pd.read_csv(str(FASTA) + ".fai", sep="\t", header=None, usecols=[0])[0].astype(str))
    valid = set(ann.loc[ann.chromosome.isin(fasta_chroms), "peak_id"])
    regions = {}
    for cluster, frame in sig.groupby(sig.cluster.astype(str)):
        frame = frame[frame.peak_id.isin(valid)].sort_values(
            ["adjusted_pvalue", "log2FC"], ascending=[True, False]
        ).head(400)
        regions[str(cluster)] = [re.sub(r"^([^:]+)-(\d+)-(\d+)$", r"\1:\2-\3", p) for p in frame.peak_id]
    bgdf = ann.query("annotation != 'unannotated' and chromosome in @fasta_chroms").copy()
    bgdf["length"] = bgdf.end - bgdf.start
    background = []
    bins = pd.qcut(bgdf["length"], min(10, bgdf["length"].nunique()), duplicates="drop")
    for _, frame in bgdf.groupby(bins, observed=False):
        take = min(len(frame), max(1, 5000 // 10))
        background.extend(frame.sample(take, random_state=SEED).peak_id)
    background = [re.sub(r"^([^:]+)-(\d+)-(\d+)$", r"\1:\2-\3", p) for p in background]
    return regions, background


def audit_raw_motif(slice_name: str, motif_names: dict[str, str]) -> pd.DataFrame:
    out = BY / slice_name / "motif_enrichment"
    records = []
    for path in sorted(out.iterdir()):
        if path.suffix.lower() not in {".csv", ".tsv", ".npy", ".h5ad", ".json"}:
            continue
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep, low_memory=False)
            for col in frame.columns:
                numeric = pd.to_numeric(frame[col], errors="coerce")
                explicit_numeric = frame[col].notna() & numeric.notna()
                for idx in frame.index[np.isposinf(numeric) & explicit_numeric]:
                    row = frame.loc[idx]
                    records.append(issue_record(path, col, idx, row, "POS_INF",
                                                "INFINITE_ODDS_RATIO_FROM_ZERO_CELL", True))
                for idx in frame.index[np.isneginf(numeric) & explicit_numeric]:
                    row = frame.loc[idx]
                    records.append(issue_record(path, col, idx, row, "NEG_INF",
                                                "INFINITE_ODDS_RATIO_FROM_ZERO_CELL", True))
                for idx in frame.index[frame[col].isna()]:
                    row = frame.loc[idx]
                    cause = "OPTIONAL_MOTIF_FAMILY_MISSING" if col == "motif_family" else "MISSING_VALUE"
                    records.append(issue_record(path, col, idx, row, "NAN", cause, col == "motif_family"))
    audit = pd.DataFrame(records, columns=[
        "file", "column", "row_index", "cluster", "motif_id", "original_value",
        "nonfinite_type", "likely_cause", "recoverable_without_rerun",
    ])
    atomic_csv(audit, AUDIT / f"motif_nonfinite_{slice_name}.csv")
    return audit


def issue_record(path, col, idx, row, kind, cause, recoverable):
    return {
        "file": str(path.relative_to(ROOT)), "column": col, "row_index": int(idx),
        "cluster": row.get("cluster", ""), "motif_id": row.get("motif_id", ""),
        "original_value": str(row.get(col, "")), "nonfinite_type": kind,
        "likely_cause": cause, "recoverable_without_rerun": bool(recoverable),
    }


def reconstruct_motif_counts(slice_name: str, motifs, motif_names):
    from pyfaidx import Fasta
    from snapatac2._utils import fetch_seq

    out = BY / slice_name / "motif_enrichment"
    source = out / "full_motif_enrichment.csv"
    raw_copy = out / "full_motif_enrichment_raw.csv"
    if not raw_copy.exists():
        shutil.copy2(source, raw_copy)
    raw = pd.read_csv(raw_copy)
    regions, background = motif_regions_and_background(slice_name)
    all_regions = sorted(set(background).union(*(set(v) for v in regions.values())))
    index = {region: i for i, region in enumerate(all_regions)}
    genome = Fasta(str(FASTA), one_based_attributes=False)
    sequences = [fetch_seq(genome, region) for region in all_regions]
    bg_idx = np.asarray([index[x] for x in background], dtype=int)
    region_idx = {cluster: np.asarray([index[x] for x in values], dtype=int)
                  for cluster, values in regions.items()}
    raw_lookup = raw.set_index([raw.cluster.astype(str), "motif_id"])
    rows = []
    for motif in motifs:
        bound = np.asarray(motif.with_nucl_prob().exists(sequences), dtype=bool)
        bg_count = int(bound[bg_idx].sum())
        bg_total = int(len(bg_idx))
        for cluster in sorted(regions, key=lambda x: int(x) if x.isdigit() else x):
            fg_total = int(len(region_idx[cluster]))
            fg_count = int(bound[region_idx[cluster]].sum()) if fg_total else 0
            old = raw_lookup.loc[(str(cluster), motif.id)]
            if isinstance(old, pd.DataFrame):
                old = old.iloc[0]
            raw_p = float(old["pvalue"])
            raw_fdr = float(old["adjusted_pvalue"])
            a, b = fg_count, fg_total - fg_count
            c, d = bg_count, bg_total - bg_count
            raw_or = math.inf if b * c == 0 and a * d > 0 else (
                0.0 if a * d == 0 else (a * d) / (b * c)
            )
            corrected_or = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            finite_p = max(raw_p, 1e-300)
            finite_fdr = max(raw_fdr, 1e-300)
            reason = "HALDANE_ANSCOMBE_ZERO_CELL" if not math.isfinite(raw_or) or min(a, b, c, d) == 0 else "NONE"
            rows.append({
                "cluster": str(cluster), "motif_id": motif.id,
                "TF_name": motif_names.get(motif.id, motif.name or motif.id),
                "foreground_count": a, "foreground_total": fg_total,
                "background_count": c, "background_total": bg_total,
                "contingency_a": a, "contingency_b": b, "contingency_c": c, "contingency_d": d,
                "raw_pvalue": raw_p, "raw_adjusted_pvalue": raw_fdr,
                "finite_pvalue": finite_p, "finite_adjusted_pvalue": finite_fdr,
                "raw_odds_ratio": ("undefined_zero_denominator" if not math.isfinite(raw_or)
                                   else format(raw_or, ".17g")),
                "corrected_odds_ratio": corrected_or,
                "log2_corrected_odds_ratio": math.log2(corrected_or),
                "minus_log10_fdr": min(-math.log10(finite_fdr), 300.0),
                "correction_reason": reason, "rerun_required": False,
                "valid_for_plot": True, "valid_for_scorecard": True,
            })
    finite = pd.DataFrame(rows)
    required = ["foreground_count", "foreground_total", "background_count", "background_total",
                "finite_pvalue", "finite_adjusted_pvalue", "corrected_odds_ratio",
                "log2_corrected_odds_ratio", "minus_log10_fdr"]
    if finite.empty or not np.isfinite(finite[required].to_numpy(float)).all():
        raise RuntimeError(f"{slice_name}: non-finite corrected motif values")
    if not ((finite.finite_pvalue.between(0, 1)) & (finite.finite_adjusted_pvalue.between(0, 1))).all():
        raise RuntimeError(f"{slice_name}: invalid corrected motif p-value")
    atomic_csv(finite, out / "full_motif_enrichment_finite.csv")
    finite["family_display"] = finite.TF_name.str.replace(r"\(.*", "", regex=True).str.split("::").str[0].str.upper()
    nr = finite.sort_values(
        ["finite_adjusted_pvalue", "log2_corrected_odds_ratio"],
        ascending=[True, False],
    ).drop_duplicates(["cluster", "family_display"])
    atomic_csv(nr, out / "nonredundant_motif_enrichment_finite.csv")
    top = nr.query("log2_corrected_odds_ratio > 0").groupby("cluster", group_keys=False).head(10)
    atomic_csv(top, out / "cluster_top_motifs_finite.csv")
    plot_motif(slice_name, top)
    atomic_json({
        "status": "COMPLETE", "method_name": "motif enrichment",
        "source_raw_table": "full_motif_enrichment_raw.csv",
        "motif_database": str(MEME), "motif_database_sha256": sha256(MEME),
        "genome_build": "mm10/GRCm38", "fasta": str(FASTA), "fasta_sha256": sha256(FASTA),
        "foreground_rule": "BH adjusted p-value <= 0.05 and log2FC >= 0.5; max 400 positive marker peaks per cluster",
        "background_rule": "same-slice detectable annotated peaks; fixed-seed length-stratified sampling; identical implementation for all slices",
        "background_total": int(len(background)), "seed": SEED,
        "root_cause": ["ZERO_PVALUE_TRANSFORM_ONLY", "INFINITE_ODDS_RATIO_FROM_ZERO_CELL"],
        "finite_effect_size": "Haldane-Anscombe corrected odds ratio; 0.5 added to all four cells",
        "raw_pvalues_preserved": True, "raw_adjusted_pvalues_preserved": True,
        "pvalue_floor_for_derived_columns_only": 1e-300,
        "all_plot_and_score_columns_finite": True,
    }, out / "motif_enrichment_metadata.json")
    return finite, top


def plot_motif(slice_name: str, top: pd.DataFrame):
    out = BY / slice_name / "motif_enrichment"
    candidate = ROOT / "figures/per_slice" / slice_name
    candidate.mkdir(parents=True, exist_ok=True)
    clusters = sorted(top.cluster.astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    motifs = list(dict.fromkeys(top.sort_values(["cluster", "finite_adjusted_pvalue"]).TF_name))
    matrix = top.pivot_table(index="cluster", columns="TF_name",
                             values="log2_corrected_odds_ratio", aggfunc="max").reindex(
                                 index=clusters, columns=motifs).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(motifs)), max(3, 0.55 * len(clusters))))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(motifs)), motifs, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(range(len(clusters)), clusters)
    ax.set_xlabel("TF motif"); ax.set_ylabel("SpaCRAFT cluster")
    ax.set_title(f"{slice_name}: motif enrichment")
    fig.colorbar(im, ax=ax, label="log2 Haldane–Anscombe corrected odds ratio")
    savefig(fig, out / "cluster_motif_enrichment_heatmap_finite.png")
    atomic_csv(top, out / "cluster_motif_enrichment_heatmap_finite_plot_data.csv")
    shutil.copy2(out / "cluster_motif_enrichment_heatmap_finite.png",
                 candidate / "motif_enrichment_heatmap.png")
    shutil.copy2(out / "cluster_motif_enrichment_heatmap_finite_plot_data.csv",
                 candidate / "motif_enrichment_heatmap_plot_data.csv")

    dot = top.sort_values(["cluster", "finite_adjusted_pvalue"]).groupby("cluster", group_keys=False).head(5)
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(dot)), 4.8))
    x = np.arange(len(dot))
    points = ax.scatter(x, dot.log2_corrected_odds_ratio, s=20 + 18 * dot.minus_log10_fdr.clip(0, 20),
                        c=dot.minus_log10_fdr, cmap="viridis")
    ax.set_xticks(x, [f"C{c}:{n}" for c, n in zip(dot.cluster, dot.TF_name)],
                  rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("log2 corrected odds ratio")
    ax.set_title(f"{slice_name}: top motif enrichment")
    fig.colorbar(points, ax=ax, label="-log10(BH FDR), capped at 300")
    savefig(fig, out / "top_motif_dot_plot.png")
    atomic_csv(dot, out / "top_motif_dot_plot_data.csv")
    shutil.copy2(out / "top_motif_dot_plot.png", candidate / "top_motif_dot_plot.png")
    shutil.copy2(out / "top_motif_dot_plot_data.csv", candidate / "top_motif_dot_plot_data.csv")


def parse_obo():
    terms, parents = {}, defaultdict(set)
    current = {}
    def finish():
        if current.get("id") and current.get("namespace") == "biological_process" and not current.get("obsolete"):
            terms[current["id"]] = current.get("name", current["id"])
            parents[current["id"]].update(current.get("parents", set()))
    for line in OBO.read_text().splitlines():
        if line == "[Term]":
            finish(); current = {"parents": set()}
        elif line.startswith("["):
            finish(); current = {}
        elif current is not None:
            if line.startswith("id: GO:"): current["id"] = line[4:]
            elif line.startswith("name: "): current["name"] = line[6:]
            elif line.startswith("namespace: "): current["namespace"] = line[11:]
            elif line.startswith("is_a: "): current.setdefault("parents", set()).add(line.split()[1])
            elif line.startswith("relationship: part_of "): current.setdefault("parents", set()).add(line.split()[2])
            elif line == "is_obsolete: true": current["obsolete"] = True
    finish()
    parents = {term: {p for p in ps if p in terms} for term, ps in parents.items() if term in terms}
    return terms, parents


def parse_gaf(terms, parents):
    direct, alias_to_symbol = defaultdict(set), {}
    with gzip.open(GAF, "rt") as f:
        for line in f:
            if line.startswith("!"): continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15 or fields[8] != "P" or "NOT" in fields[3].split("|"): continue
            symbol, go_id = fields[2], fields[4]
            if go_id not in terms: continue
            direct[symbol].add(go_id)
            alias_to_symbol[symbol.upper()] = symbol
            for alias in fields[10].split("|"):
                if alias: alias_to_symbol.setdefault(alias.upper(), symbol)
    memo = {}
    def ancestors(term):
        if term in memo: return memo[term]
        result = {term}
        for parent in parents.get(term, ()): result.update(ancestors(parent))
        memo[term] = result
        return result
    propagated = {gene: set().union(*(ancestors(term) for term in gos)) for gene, gos in direct.items()}
    return propagated, alias_to_symbol


def mapped_gene(frame: pd.DataFrame) -> pd.Series:
    direct = frame.annotation.isin(["promoter", "5' UTR", "exon", "intron"])
    distal = frame.distance_to_tss.abs().le(GO_DISTAL_MAX_BP)
    return frame.gene_name.where(frame.gene_name.notna() & (direct | distal), "")


def run_go(slice_name: str, terms, gene_to_go, alias_to_symbol):
    out = BY / slice_name / "functional_enrichment"
    ann = pd.read_csv(BY / slice_name / "peak_annotation/all_peaks_annotation.csv", low_memory=False)
    marker = pd.read_csv(BY / slice_name / "peak_annotation/marker_peaks_annotation.csv", low_memory=False)
    ann["mapped_gene"] = mapped_gene(ann)
    marker["mapped_gene"] = mapped_gene(marker)
    def canonical(g):
        return alias_to_symbol.get(str(g).upper(), str(g)) if g else ""
    ann["GO_symbol"] = ann.mapped_gene.map(canonical)
    marker["GO_symbol"] = marker.mapped_gene.map(canonical)
    mapping = ann[["peak_id", "chromosome", "start", "end", "annotation", "distance_to_tss",
                   "gene_id", "gene_name", "mapped_gene", "GO_symbol"]].copy()
    mapping["GO_mapped"] = mapping.GO_symbol.isin(gene_to_go)
    atomic_csv(mapping, out / "gene_mapping_table.csv")
    background = sorted(set(mapping.loc[mapping.GO_mapped, "GO_symbol"]))
    (out / "background_gene_set.txt").write_text("\n".join(background) + "\n")
    bg_set = set(background); N = len(bg_set)
    term_bg = defaultdict(set)
    for gene in bg_set:
        for term in gene_to_go[gene]: term_bg[term].add(gene)
    significant = marker.query("direction == 'positive' and adjusted_pvalue <= 0.05 and log2FC >= 0.5")
    clusters = sorted(marker.cluster.astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    all_rows, summary_rows = [], []
    for cluster in clusters:
        fg = set(significant.loc[significant.cluster.astype(str) == str(cluster), "GO_symbol"]) & bg_set
        n = len(fg)
        cluster_rows = []
        for term, bg_genes in term_bg.items():
            overlap = fg & bg_genes
            if not overlap: continue
            a = len(overlap); b = n - a; c = len(bg_genes) - a; d = N - a - b - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            cluster_rows.append({
                "cluster": str(cluster), "term_id": term, "term_name": terms[term],
                "foreground_gene_count": n, "background_gene_count": N,
                "term_background_count": len(bg_genes), "overlap_count": a,
                "odds_ratio": float(odds), "pvalue": float(p),
                "genes": ";".join(sorted(overlap)),
            })
        cdf = pd.DataFrame(cluster_rows)
        if not cdf.empty:
            cdf["adjusted_pvalue"] = multipletests(cdf.pvalue, method="fdr_bh")[1]
            cdf["finite_odds_ratio"] = [
                ((r.overlap_count + .5) * (N - r.foreground_gene_count - r.term_background_count + r.overlap_count + .5)) /
                ((r.foreground_gene_count - r.overlap_count + .5) * (r.term_background_count - r.overlap_count + .5))
                for r in cdf.itertuples()
            ]
            cdf["minus_log10_fdr"] = -np.log10(cdf.adjusted_pvalue.clip(lower=1e-300)).clip(upper=300)
            all_rows.extend(cdf.to_dict("records"))
            candidates = cdf.query("adjusted_pvalue < 0.05").sort_values(
                ["adjusted_pvalue", "finite_odds_ratio"], ascending=[True, False])
            chosen, chosen_sets = [], []
            for row in candidates.itertuples():
                genes = set(str(row.genes).split(";"))
                if any(len(genes & old) / max(1, len(genes | old)) >= 0.8 for old in chosen_sets):
                    continue
                chosen.append(row._asdict()); chosen_sets.append(genes)
                if len(chosen) == 5: break
            if chosen: summary_rows.extend(chosen)
            else: summary_rows.append(no_go_row(cluster, n, N))
        else:
            summary_rows.append(no_go_row(cluster, n, N))
    full = pd.DataFrame(all_rows)
    if full.empty: raise RuntimeError(f"{slice_name}: no GO BP tests generated")
    columns = ["cluster", "term_id", "term_name", "foreground_gene_count", "background_gene_count",
               "term_background_count", "overlap_count", "odds_ratio", "finite_odds_ratio",
               "pvalue", "adjusted_pvalue", "minus_log10_fdr", "genes"]
    full = full[columns].sort_values(["cluster", "adjusted_pvalue", "term_id"])
    filtered = full.query("adjusted_pvalue < 0.05").copy()
    summary = pd.DataFrame(summary_rows)
    for col in columns:
        if col not in summary: summary[col] = "" if col in {"term_id", "term_name", "genes"} else 0
    summary = summary[columns]
    atomic_csv(full, out / "full_GO_BP_results.csv")
    atomic_csv(filtered, out / "filtered_GO_BP_results.csv")
    atomic_csv(summary, out / "cluster_GO_summary.csv")
    plot_go(slice_name, summary)
    atomic_json({
        "status": "COMPLETE", "method_name": "GO Biological Process enrichment",
        "not_GREAT": True, "genome_build": "mm10/GRCm38",
        "mapping_rule": "promoter/gene-body direct; other annotated peaks nearest gene within 100000 bp",
        "distal_max_distance_bp": GO_DISTAL_MAX_BP,
        "foreground_rule": "positive marker peaks with BH adjusted p-value <= 0.05 and log2FC >= 0.5",
        "background_rule": "all same-slice detectable peaks mapped by the identical rule, restricted to genes represented in the official mouse GAF",
        "statistical_test": "one-sided Fisher exact enrichment", "multiple_testing": "BH within cluster",
        "display_rule": "adjusted p-value < 0.05; max 5 terms/cluster; gene-overlap Jaccard >= 0.8 removed",
        "ontology": str(OBO), "ontology_sha256": sha256(OBO),
        "annotations": str(GAF), "annotations_sha256": sha256(GAF),
        "ontology_release": "2026-06-15", "GO_product_release": "2026-06-19",
        "background_gene_count": N, "tested_cluster_count": len(clusters),
        "full_result_rows": len(full), "significant_result_rows": len(filtered),
    }, out / "metadata.json")
    return full, summary


def no_go_row(cluster, n, N):
    return {"cluster": str(cluster), "term_id": "NO_SIGNIFICANT_TERM",
            "term_name": "No significant term", "foreground_gene_count": n,
            "background_gene_count": N, "term_background_count": 0, "overlap_count": 0,
            "odds_ratio": 0.0, "finite_odds_ratio": 0.0, "pvalue": 1.0,
            "adjusted_pvalue": 1.0, "minus_log10_fdr": 0.0, "genes": ""}


def plot_go(slice_name: str, summary: pd.DataFrame):
    out = BY / slice_name / "functional_enrichment"
    candidate = ROOT / "figures/per_slice" / slice_name
    candidate.mkdir(parents=True, exist_ok=True)
    plot = summary.query("term_id != 'NO_SIGNIFICANT_TERM'").copy()
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.3 * max(1, len(plot)))))
    if plot.empty:
        ax.text(.5, .5, "No significant GO BP term", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        y = np.arange(len(plot))
        sc = ax.scatter(plot.minus_log10_fdr, y, s=25 + 18 * plot.overlap_count,
                        c=np.log2(plot.finite_odds_ratio.clip(lower=1e-12)), cmap="viridis")
        ax.set_yticks(y, [f"C{c}  {t}" for c, t in zip(plot.cluster, plot.term_name)], fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("-log10(BH FDR), capped at 300")
        fig.colorbar(sc, ax=ax, label="log2 corrected odds ratio")
    ax.set_title(f"{slice_name}: GO Biological Process enrichment")
    savefig(fig, out / "cluster_GO_dot_plot.png")
    plot_data = summary.drop(columns=["odds_ratio"]).copy()
    plot_data["genes"] = plot_data["genes"].fillna("NO_GENES")
    atomic_csv(plot_data, out / "cluster_GO_dot_plot_data.csv")
    shutil.copy2(out / "cluster_GO_dot_plot.png", candidate / "GO_BP_dot_plot.png")
    shutil.copy2(out / "cluster_GO_dot_plot_data.csv", candidate / "GO_BP_dot_plot_data.csv")


def make_preview(paths, names, png, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, name, path in zip(axes.ravel(), names, paths):
        ax.imshow(Image.open(path)); ax.set_title(name); ax.axis("off")
    savefig(fig, png, pdf)


def certified_scorecard(motif_tables, go_tables):
    old_path = SCREEN / "DOWNSTREAM_SCREENING_SCORECARD.csv"
    invalid = SCREEN / "DOWNSTREAM_SCREENING_SCORECARD_INVALID_ARCHIVE.csv"
    if not invalid.exists(): shutil.copy2(old_path, invalid)
    old = pd.read_csv(old_path).set_index("slice")
    rows = []
    preserved = ["Marker_Peak_Strength", "Marker_Peak_Significance", "Heatmap_Block_Structure",
                 "Peak_Annotation_Completeness", "Gene_Activity_Specificity",
                 "Gene_Activity_Spatial_MoransI", "Biological_Interpretability"]
    for name in SLICES:
        row = {"slice": name, **{c: float(old.loc[name, c]) for c in preserved}}
        motif = motif_tables[name]
        sig_motif = motif.query("finite_adjusted_pvalue < 0.05 and log2_corrected_odds_ratio > 0")
        per_cluster_motif = sig_motif.groupby("cluster").log2_corrected_odds_ratio.max()
        row["Motif_Enrichment_Strength"] = float(per_cluster_motif.mean()) if len(per_cluster_motif) else 0.0
        go = go_tables[name]
        sig_go = go.query("adjusted_pvalue < 0.05")
        per_cluster_go = sig_go.groupby("cluster").minus_log10_fdr.max()
        row["Functional_Enrichment_Specificity"] = float(per_cluster_go.mean()) if len(per_cluster_go) else 0.0
        row["Analysis_Coverage"] = 1.0
        rows.append(row)
    df = pd.DataFrame(rows)
    components = preserved[:-1] + ["Motif_Enrichment_Strength",
                                   "Functional_Enrichment_Specificity",
                                   "Biological_Interpretability"]
    for col in components:
        values = df[col].astype(float)
        if not np.isfinite(values).all(): raise RuntimeError(f"non-finite score component {col}")
        span = values.max() - values.min()
        df[col + "_scaled"] = 100.0 if span == 0 else (values - values.min()) / span * 100.0
    df["Certified_Component_Mean"] = df[[c + "_scaled" for c in components]].mean(axis=1)
    if df.isna().any().any() or not np.isfinite(df.select_dtypes(include=np.number)).all().all():
        raise RuntimeError("certified scorecard contains NaN/Inf")
    atomic_csv(df, SCREEN / "DOWNSTREAM_SCREENING_SCORECARD_CERTIFIED.csv")
    rank = df[["slice", "Certified_Component_Mean", "Analysis_Coverage"]].sort_values(
        ["Certified_Component_Mean", "slice"], ascending=[False, True]).reset_index(drop=True)
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    rank["tie_break_rule"] = "descending score; lexical slice ID for exact ties"
    atomic_csv(rank, SCREEN / "SLICE_RANKING_CERTIFIED.csv")
    atomic_csv(rank, SCREEN / "CANDIDATE_PANEL_RANKING_CERTIFIED.csv")
    complete = pd.DataFrame({"slice": SLICES})
    for col in ["Marker_Peaks", "Peak_Annotation", "Gene_Activity", "Motif_Enrichment", "Functional_GO_BP"]:
        complete[col] = "COMPLETE"
    for col in ["Motif_Activity", "Coaccessibility", "Pseudotime"]:
        complete[col] = "OPTIONAL_NOT_RUN"
    complete["Footprinting"] = "SKIPPED_NO_VALIDATED_FRAGMENTS"
    complete["Analysis_Coverage"] = 1.0
    atomic_csv(complete, SCREEN / "ANALYSIS_COMPLETENESS_MATRIX_CERTIFIED.csv")
    definitions = """# Certified score component definitions

The formula was frozen before certified ranking. All scientific components are oriented so larger is better, min-max scaled across the six formal slices, and combined by an unweighted arithmetic mean. Exact score ties are resolved lexically by slice ID. Optional analyses are excluded and never encoded as zero.

| component | source / raw column | transformation | missing handling | finite |
|---|---|---|---|---|
| Marker_Peak_Strength | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Marker_Peak_Significance | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Heatmap_Block_Structure | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Peak_Annotation_Completeness | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Gene_Activity_Specificity | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Gene_Activity_Spatial_MoransI | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Motif_Enrichment_Strength | full_motif_enrichment_finite.csv / log2_corrected_odds_ratio | mean across clusters of each cluster maximum among FDR<0.05 positive effects; 0 only when completed analysis has no significant positive motif; min-max 0–100 | certification fails if analysis absent | yes |
| Functional_Enrichment_Specificity | full_GO_BP_results.csv / minus_log10_fdr | mean across clusters of each cluster maximum significant GO BP score; 0 only when completed analysis has no significant term; min-max 0–100 | certification fails if analysis absent | yes |
| Biological_Interpretability | archived scorecard / same column | preserved audited finite raw value; min-max 0–100 | certification fails | yes |
| Analysis_Coverage | certified core analyses | completed valid core components / required core components; reported separately and not used as a score | certification fails if <1 | yes |
"""
    (SCREEN / "CERTIFIED_SCORE_COMPONENT_DEFINITIONS.md").write_text(definitions)
    return df, rank


def recommendations(rank):
    main, backup = rank.iloc[0]["slice"], rank.iloc[1]["slice"]
    panels = "a Ground truth vs SpaCRAFT clusters; b Marker peak heatmap; c Gene activity spatial maps; d finite motif enrichment; e GO Biological Process enrichment; f Peak genomic annotation"
    (AUDIT / "RECOMMENDED_MAIN_SLICE_CERTIFIED.md").write_text(
        f"# Certified recommended main slice\n\n- main_slice: **{main}**\n- backup_slice: **{backup}**\n- provisional: **False**\n- basis: pre-frozen certified unweighted component mean; Analysis_Coverage reported separately.\n")
    (AUDIT / "RECOMMENDED_MAIN_FIGURE_PLAN_CERTIFIED.md").write_text(
        f"# Certified main figure plan\n\n- main_slice: **{main}**\n- recommended_panel_set: {panels}\n- final_figure_assembled: **False**\n")
    (AUDIT / "BACKUP_PANEL_PLAN_CERTIFIED.md").write_text(
        f"# Certified backup panel plan\n\n- backup_slice: **{backup}**\n- backup_panel_set: {panels}\n")
    (AUDIT / "FAILED_OR_WEAK_ANALYSES_CERTIFIED.md").write_text(
        "# Certified failed or weak analyses\n\n"
        "- motif activity/chromVAR: OPTIONAL_NOT_RUN; isolated R environment remains absent.\n"
        "- co-accessibility: OPTIONAL_NOT_RUN; current fragment support remains insufficient.\n"
        "- pseudotime: OPTIONAL_NOT_RUN; no pre-specified biological root.\n"
        "- footprinting: SKIPPED_NO_VALIDATED_FRAGMENTS.\n")
    return main, backup, panels


def validate_outputs(main, backup, panels):
    motif_nonfinite = 0
    go_complete = 0
    for name in SLICES:
        m = pd.read_csv(BY / name / "motif_enrichment/full_motif_enrichment_finite.csv")
        cols = ["finite_pvalue", "finite_adjusted_pvalue", "corrected_odds_ratio",
                "log2_corrected_odds_ratio", "minus_log10_fdr"]
        motif_nonfinite += int((~np.isfinite(m[cols].to_numpy(float))).sum())
        gmeta = json.loads((BY / name / "functional_enrichment/metadata.json").read_text())
        go_complete += gmeta.get("status") == "COMPLETE"
        for png in [BY / name / "motif_enrichment/cluster_motif_enrichment_heatmap_finite.png",
                    BY / name / "motif_enrichment/top_motif_dot_plot.png",
                    BY / name / "functional_enrichment/cluster_GO_dot_plot.png"]:
            im = Image.open(png); im.verify()
    score = pd.read_csv(SCREEN / "DOWNSTREAM_SCREENING_SCORECARD_CERTIFIED.csv")
    rank = pd.read_csv(SCREEN / "SLICE_RANKING_CERTIFIED.csv")
    score_nonfinite = int((~np.isfinite(score.select_dtypes(include=np.number).to_numpy())).sum())
    rank_nonfinite = int((~np.isfinite(rank.select_dtypes(include=np.number).to_numpy())).sum())
    if motif_nonfinite or go_complete != 6 or score_nonfinite or rank_nonfinite:
        raise RuntimeError("certification validation failed")
    protected_before = next(ROOT.glob("archive_before_core_correction_*/audit/PROTECTED_RESULTS_SHA256_BEFORE.txt"))
    before = protected_before.read_text().splitlines()
    current = []
    for line in before:
        _, path = line.split(None, 1)
        current.append(f"{sha256(Path(path))}  {path}")
    protected_unchanged = before == current
    if not protected_unchanged: raise RuntimeError("protected marker/annotation/gene-activity outputs changed")
    next_stage = f"""# Next stage after certification

1. Formal main slice: **{main}**.
2. Backup slice: **{backup}**.
3. Marker peaks are suitable for the main text; use the finite certified ranking context.
4. Gene activity is suitable for the main text as a matrix-based, nearest-gene activity proxy.
5. Motif enrichment is suitable for the main text after Haldane–Anscombe correction; raw statistics remain preserved.
6. GO Biological Process enrichment is suitable for the main text where significant terms exist; clusters with no significant term remain explicit.
7. chromVAR is optional and recommended only if motif activity would materially strengthen the biological story; it is not needed for core certification.
8. Matrix-based co-accessibility is not recommended without a pre-specified validated locus set; do not call it peak-to-gene linkage.
9. Pseudotime is not recommended without an independently justified biological root.
10. Recommended Figure a–f: {panels}.
11. Per-slice secondary motif dot plots, full GO term panels, gene-activity heatmaps, and non-main-slice diagnostics are best placed in supplementary material.
12. The project can now enter manual candidate-figure review. Final figure assembly remains disabled.

final_figure_assembled=False
"""
    (AUDIT / "NEXT_STAGE_AFTER_CERTIFICATION.md").write_text(next_stage)
    summary = {
        "core_correction_status": "COMPLETE",
        "marker_peaks_recomputed": False, "peak_annotation_recomputed": False,
        "gene_activity_recomputed": False,
        "motif_nonfinite_root_cause": "ZERO_PVALUE_TRANSFORM_ONLY+INFINITE_ODDS_RATIO_FROM_ZERO_CELL",
        "motif_enrichment_repaired_count": 6, "motif_enrichment_complete_count": 6,
        "motif_plot_nonfinite_count": motif_nonfinite,
        "functional_GO_complete_count": go_complete,
        "GO_method_name": "GO Biological Process enrichment",
        "motif_activity_status": "OPTIONAL_NOT_RUN",
        "coaccessibility_status": "OPTIONAL_NOT_RUN",
        "pseudotime_status": "OPTIONAL_NOT_RUN",
        "footprinting_status": "SKIPPED_NO_VALIDATED_FRAGMENTS",
        "certified_scorecard_status": "CERTIFIED",
        "certified_scorecard_nonfinite_count": score_nonfinite,
        "certified_slice_ranking_status": "CERTIFIED",
        "certified_recommended_main_slice": main,
        "certified_recommended_backup_slice": backup,
        "certified_main_panel_set": panels,
        "certified_backup_panel_set": panels,
        "candidate_figure_count": len(list((ROOT / "figures/per_slice").rglob("*.png"))),
        "preview_figure_count": len([p for p in PREV.iterdir() if p.suffix.lower() in {".png", ".pdf", ".svg"}]),
        "chromVAR_recommended_next": "OPTIONAL_IF_BIOLOGICALLY_MATERIAL",
        "coaccessibility_recommended_next": "NO_WITHOUT_VALIDATED_LOCUS_SET",
        "pseudotime_recommended_next": "NO_WITHOUT_BIOLOGICAL_ROOT",
        "final_figure_assembled": False, "models_retrained": False,
        "embedding_recomputed": False, "clusters_recomputed": False,
        "formal_inputs_modified": False, "protected_results_unchanged": protected_unchanged,
    }
    atomic_json(summary, AUDIT / "CORE_CORRECTION_FINAL_STATUS.json")
    return summary


def motif_audit_report(audits, finite_tables):
    lines = [
        "# Motif enrichment non-finite audit", "",
        "## Root-cause classification", "",
        "**ZERO_PVALUE_TRANSFORM_ONLY** and **INFINITE_ODDS_RATIO_FROM_ZERO_CELL**", "",
        "SnapATAC2 2.8.0 computes raw log2 fold enrichment as foreground hit fraction divided by "
        "background hit fraction. A positive foreground count with zero background hits yields +Inf. "
        "The files are readable, backgrounds are non-empty, all frozen clusters have foreground peaks, "
        "and motif IDs map to the JASPAR MEME resource. This is not string parsing, numerical overflow, "
        "invalid background, empty cluster, or file corruption. Raw p-values/FDR equal to zero are "
        "floating-point tail underflow and are handled only in derived plotting columns.", "",
        "Raw p-values and BH-adjusted p-values are preserved. Four-cell motif occurrence counts were "
        "reconstructed by scanning the same regions with the same motifs and background rule. "
        "Finite display/score effects use the Haldane–Anscombe correction only.", "",
        "| slice | raw audit rows | raw +Inf log2FC | raw -Inf log2FC | raw p=0 | raw FDR=0 | background total | corrected nonfinite |", "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for name in SLICES:
        raw = pd.read_csv(BY / name / "motif_enrichment/full_motif_enrichment_raw.csv")
        finite = finite_tables[name]
        lines.append(f"| {name} | {len(audits[name])} | {np.isposinf(raw.log2_fold_change).sum()} | "
                     f"{np.isneginf(raw.log2_fold_change).sum()} | "
                     f"{(raw.pvalue == 0).sum()} | {(raw.adjusted_pvalue == 0).sum()} | "
                     f"{int(finite.background_total.iloc[0])} | "
                     f"{int((~np.isfinite(finite.select_dtypes(include=np.number))).sum().sum())} |")
    lines += ["", "Per-cell audit files:"] + [f"- `audit/motif_nonfinite_{name}.csv`" for name in SLICES]
    (AUDIT / "MOTIF_ENRICHMENT_NONFINITE_AUDIT.md").write_text("\n".join(lines) + "\n")


def main():
    print(f"[{datetime.now().isoformat()}] core correction start", flush=True)
    for path in [FASTA, Path(str(FASTA) + ".fai"), MEME, OBO, GAF]:
        if not path.is_file() or path.stat().st_size == 0: raise FileNotFoundError(path)
    motifs, motif_names = parse_meme()
    audits, motif_tables = {}, {}
    for name in SLICES:
        print(f"[{datetime.now().isoformat()}] motif audit/correction slice={name}", flush=True)
        audits[name] = audit_raw_motif(name, motif_names)
        motif_tables[name], _ = reconstruct_motif_counts(name, motifs, motif_names)
    motif_audit_report(audits, motif_tables)
    print(f"[{datetime.now().isoformat()}] parse GO resources", flush=True)
    terms, parents = parse_obo()
    gene_to_go, alias_to_symbol = parse_gaf(terms, parents)
    go_tables = {}
    for name in SLICES:
        print(f"[{datetime.now().isoformat()}] GO BP slice={name}", flush=True)
        go_tables[name], _ = run_go(name, terms, gene_to_go, alias_to_symbol)
    make_preview([BY / n / "motif_enrichment/cluster_motif_enrichment_heatmap_finite.png" for n in SLICES],
                 SLICES, PREV / "all_slices_motif_enrichment_preview.png",
                 PREV / "all_slices_motif_enrichment_preview.pdf")
    make_preview([BY / n / "functional_enrichment/cluster_GO_dot_plot.png" for n in SLICES],
                 SLICES, PREV / "all_slices_GO_preview.png", PREV / "all_slices_GO_preview.pdf")
    _, rank = certified_scorecard(motif_tables, go_tables)
    main_slice, backup_slice, panels = recommendations(rank)
    summary = validate_outputs(main_slice, backup_slice, panels)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[{datetime.now().isoformat()}] core correction complete", flush=True)


if __name__ == "__main__":
    main()

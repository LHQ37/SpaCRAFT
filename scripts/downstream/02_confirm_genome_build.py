#!/usr/bin/env python3
"""Parse and audit all peak coordinates; validate bounds when sizes exist."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import re
import pandas as pd
import anndata as ad

ROOT = SINGLE_DOWNSTREAM_ROOT
manifest = pd.read_csv(ROOT / "audit/SINGLE_SLICE_INPUT_MANIFEST.csv")
sizes_path = GENOME_ROOT / "genome.chrom.sizes"
sizes = {}
if sizes_path.is_file():
    sizes = dict(pd.read_csv(sizes_path, sep="\t", header=None, names=["chrom", "size"]).values)
aliases = {}
alias_path = GENOME_ROOT / "chromosome_alias.tsv"
if alias_path.is_file():
    for line in alias_path.read_text().splitlines():
        row = line.split("\t")
        ucsc = row[0]
        for value in row:
            if value:
                aliases[value] = ucsc
pat = re.compile(r"^([^:-]+)[:-](\d+)[:-](\d+)$")
rows = []
peak_sets = {}
for r in manifest.itertuples():
    a = ad.read_h5ad(r.source_h5ad, backed="r")
    peaks = a.var_names.astype(str)
    peak_sets[r.dataset_name] = set(peaks)
    parsed = [pat.match(x) for x in peaks]
    bad_parse = sum(x is None for x in parsed)
    bad_interval = negative = out = 0
    chroms = set()
    lengths = []
    for m in parsed:
        if m is None:
            continue
        chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        chroms.add(chrom); lengths.append(end - start)
        negative += start < 0
        bad_interval += start >= end
        size_chrom = chrom if chrom in sizes else aliases.get(chrom, chrom)
        out += bool(sizes and (size_chrom not in sizes or end > int(sizes.get(size_chrom, -1))))
    rows.append(dict(dataset_name=r.dataset_name, peak_count=len(peaks),
        parsed_count=len(peaks)-bad_parse, parse_failures=bad_parse,
        duplicate_peaks=int(pd.Index(peaks).duplicated().sum()),
        negative_coordinates=negative, start_ge_end=bad_interval,
        out_of_bounds=out if sizes else "NOT_YET_TESTED",
        chromosome_naming="chr-prefixed" if all(x.startswith("chr") or x.startswith("GL") or x.startswith("JH") for x in chroms) else "mixed",
        chrM_present="chrM" in chroms, interval_length_min=min(lengths), interval_length_max=max(lengths),
        coordinate_convention="0-based half-open (source BED/peak convention)",
        genome_build="mm10/GRCm38"))
out_df = pd.DataFrame(rows)
out_df.to_csv(ROOT / "audit/PEAK_COORDINATE_DETAILS.csv", index=False)
all_ok = (out_df.parse_failures.eq(0) & out_df.duplicate_peaks.eq(0) &
          out_df.negative_coordinates.eq(0) & out_df.start_ge_end.eq(0)).all()
if sizes:
    all_ok = all_ok and out_df.out_of_bounds.astype(int).eq(0).all()
pairwise_same = len({frozenset(x) for x in peak_sets.values()}) == 1
table = ["| " + " | ".join(map(str, out_df.columns)) + " |",
         "|" + "|".join(["---"] * len(out_df.columns)) + "|"]
table.extend("| " + " | ".join(map(str, row)) + " |" for row in out_df.itertuples(index=False, name=None))
lines = ["# Peak coordinate audit", "",
    f"- Status: **{'PASS' if all_ok else 'FAILED'}**",
    "- Formats: colon-delimited (`chr:start-end`) in four MISAR-seq slices; hyphen-delimited (`chr-start-end`) in the P22 and E12.5 inputs.",
    "- Coordinates are interpreted as source peak/BED 0-based half-open intervals.",
    f"- Six peak sets identical: **{pairwise_same}** (expected false; each slice retains its source peak set).",
    f"- Chromosome bounds tested: **{bool(sizes)}**", "",
    *table, ""]
(ROOT / "audit/PEAK_COORDINATE_AUDIT.md").write_text("\n".join(lines))
print("peak_coordinate_status=" + ("PASS" if all_ok else "FAILED"))

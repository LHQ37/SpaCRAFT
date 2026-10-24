#!/usr/bin/env python3
"""Map Phase-3 module peaks with the frozen legal mapping and describe direction concordance."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import pandas as pd

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/gene_activity_phase4_v1"; MOD=ROOT/"results/accessibility_phase3_v1/modules"
DEST=OUT/"module_concordance"; META=OUT/"metadata"
pm=pd.read_csv(META/"primary_peak_to_gene_mapping.csv.gz")
ga=pd.read_csv(OUT/"C5/C5_ranked_gene_activity_dynamics.csv")
members=pd.read_csv(MOD/"C5_all_modules_membership.csv")
peakcol=next(c for c in members if "peak" in c.lower())
modulecol=next(c for c in members if "module" in c.lower())
x=members[[peakcol,modulecol]].rename(columns={peakcol:"peak_id",modulecol:"module"}).merge(
    pm[["peak_id","gene_id","gene_symbol"]],on="peak_id",how="left")
x.to_csv(DEST/"C5_module_peak_to_gene_mapping.csv",index=False)
ug=x.dropna(subset=["gene_id"]).drop_duplicates(["module","gene_id"])[["module","gene_id","gene_symbol"]]
ug=ug.merge(ga[["gene_id","tested","pattern","late_minus_early_delta","rank_score"]],on="gene_id",how="left")
def expected(m): return "EarlyHigh" if "Early" in str(m) else "LateHigh"
ug["expected_gene_pattern"]=ug.module.map(expected)
ug["concordance_class"]="not_tested"
ug.loc[ug.tested.eq(True),"concordance_class"]="tested_other"
ug.loc[ug.tested.eq(True)&ug.pattern.eq(ug.expected_gene_pattern),"concordance_class"]="concordant"
opp={"EarlyHigh":"LateHigh","LateHigh":"EarlyHigh"}
ug.loc[ug.tested.eq(True)&ug.pattern.eq(ug.expected_gene_pattern.map(opp)),"concordance_class"]="opposite"
ug.to_csv(DEST/"C5_module_associated_genes.csv",index=False)
for pat in ("EarlyHigh","LateHigh"):
    ug[ug.expected_gene_pattern.eq(pat)].to_csv(DEST/f"C5_{pat}_module_gene_activity_concordance.csv",index=False)
rows=[]
for m,z in ug.groupby("module"):
    n=int(z.tested.eq(True).sum())
    rows.append({"module":m,"expected_pattern":expected(m),"associated_gene_count":len(z),"tested_gene_count":n,
      "concordant_count":int(z.concordance_class.eq("concordant").sum()),
      "opposite_count":int(z.concordance_class.eq("opposite").sum()),
      "concordance_fraction_among_tested":z.concordance_class.eq("concordant").sum()/max(1,n),
      "opposite_fraction_among_tested":z.concordance_class.eq("opposite").sum()/max(1,n)})
s=pd.DataFrame(rows); s.to_csv(DEST/"C5_module_gene_activity_concordance.csv",index=False)
s.to_csv(DEST/"C5_module_gene_activity_summary.csv",index=False)
# Required 8,282-axis descriptive module-direction comparison.
ax=pd.read_csv(OUT/"sensitivity/GENE_ACTIVITY_PEAK_AXIS_SENSITIVITY.csv")
axpat=ax[["gene_id","pattern_8282"]]
axs=ug.merge(axpat,on="gene_id",how="left")
axis_rows=[]
for m,z in axs.groupby("module"):
    expected_pat=expected(m); available=z.pattern_8282.notna()
    axis_rows.append({"module":m,"expected_pattern":expected_pat,"common_tested_gene_count":int(available.sum()),
      "sensitivity_axis_concordant_count":int((z.pattern_8282.eq(expected_pat)&available).sum()),
      "sensitivity_axis_direction_concordance":float((z.pattern_8282.eq(expected_pat)&available).sum()/max(1,available.sum()))})
pd.DataFrame(axis_rows).to_csv(OUT/"sensitivity/C5_module_associated_gene_direction_sensitivity.csv",index=False)
high=(s.concordance_fraction_among_tested.ge(.6)&s.opposite_fraction_among_tested.le(.2)).all()
moderate=s.concordance_fraction_among_tested.ge(.4).all()
status="HIGH_CONCORDANCE" if high else ("MODERATE_CONCORDANCE" if moderate else "LOW_CONCORDANCE")
lines=["# Module–gene-activity concordance","",
 f"Pre-registered descriptive status: `{status}`. No enrichment test or P-value was performed.",""]
for z in rows:
    lines.append(f"- {z['module']}: {z['concordant_count']}/{z['tested_gene_count']} concordant "
                 f"({z['concordance_fraction_among_tested']:.3f}); "
                 f"{z['opposite_count']} opposite ({z['opposite_fraction_among_tested']:.3f}).")
(DEST/"C5_module_gene_activity_concordance.md").write_text("\n".join(lines)+"\n")
(META/"module_gene_concordance_summary.json").write_text(json.dumps({"status":status,"modules":rows,
 "sensitivity_axis_modules":axis_rows},indent=2)+"\n")
print(status)

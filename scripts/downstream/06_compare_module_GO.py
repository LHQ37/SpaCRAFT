#!/usr/bin/env python3
"""GO redundancy, Early/Late comparison, mapping sensitivity, and quality."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/GO_phase6_v1"
COMP=OUT/"comparison"; SENS=OUT/"sensitivity"; META=OUT/"metadata"; MAP=OUT/"gene_mapping"
full={m:pd.read_csv(OUT/m/"full_GO_BP_finite.csv") for m in ("EarlyHigh","LateHigh")}
strict={m:pd.read_csv(SENS/f"{m}_promoter_gene_body_only_GO_BP.csv") for m in ("EarlyHigh","LateHigh")}
groups=[]; nr={}
for m,x in full.items():
 cand=x[x.enriched.astype(bool)].sort_values(["BH_adjusted_pvalue","foreground_gene_count","GO_ID"],ascending=[True,False,True])
 reps=[]; sets=[]; group_id=[]
 for z in cand.itertuples():
  gs=set(str(z.genes).split(";")); hit=None
  for i,old in enumerate(sets):
   if len(gs&old)/max(1,len(gs|old))>=.8: hit=i; break
  if hit is None: hit=len(reps); reps.append(z); sets.append(gs)
  groups.append({"module":m,"redundancy_group":f"{m}_G{hit+1:03d}","GO_ID":z.GO_ID,
   "representative_GO_ID":reps[hit].GO_ID,"is_representative":z.GO_ID==reps[hit].GO_ID,
   "foreground_gene_jaccard_to_representative":len(gs&sets[hit])/max(1,len(gs|sets[hit]))})
 n=pd.DataFrame([z._asdict() for z in reps])
 if n.empty: n=pd.DataFrame(columns=x.columns)
 nr[m]=n; n.to_csv(OUT/m/"nonredundant_GO_BP.csv",index=False)
pd.DataFrame(groups,columns=["module","redundancy_group","GO_ID","representative_GO_ID","is_representative","foreground_gene_jaccard_to_representative"]).to_csv(COMP/"GO_redundancy_groups.csv",index=False)

cols=["GO_ID","term","log2_corrected_odds_ratio","BH_adjusted_pvalue","foreground_gene_count","genes","enriched"]
a=full["EarlyHigh"][cols].rename(columns={c:f"EarlyHigh_{c}" for c in cols if c not in ("GO_ID","term")})
b=full["LateHigh"][cols].rename(columns={c:f"LateHigh_{c}" for c in cols if c not in ("GO_ID","term")})
x=a.merge(b,on=["GO_ID","term"],how="outer")
for c in x.columns:
 if c.startswith(("EarlyHigh_","LateHigh_")) and c.endswith(("log2_corrected_odds_ratio","foreground_gene_count")): x[c]=x[c].fillna(0)
 if c.endswith("BH_adjusted_pvalue"): x[c]=x[c].fillna(1)
 if c.endswith("_enriched"): x[c]=x[c].fillna(False).astype(bool)
x["EarlyHigh_log2OR"]=x.EarlyHigh_log2_corrected_odds_ratio; x["EarlyHigh_FDR"]=x.EarlyHigh_BH_adjusted_pvalue
x["LateHigh_log2OR"]=x.LateHigh_log2_corrected_odds_ratio; x["LateHigh_FDR"]=x.LateHigh_BH_adjusted_pvalue
x["EarlyHigh_gene_count"]=x.EarlyHigh_foreground_gene_count.astype(int); x["LateHigh_gene_count"]=x.LateHigh_foreground_gene_count.astype(int)
def gs(v): return set(str(v).split(";")) if pd.notna(v) and str(v) else set()
x["overlap_gene_count"]=[len(gs(e)&gs(l)) for e,l in zip(x.EarlyHigh_genes,x.LateHigh_genes)]
ee=x.EarlyHigh_enriched; le=x.LateHigh_enriched; same=np.sign(x.EarlyHigh_log2OR)==np.sign(x.LateHigh_log2OR)
x["category"]=np.select([ee&le,ee&~le,~ee&le,~ee&~le&same],["SHARED_ENRICHED","EARLY_SPECIFIC","LATE_SPECIFIC","SHARED_NOT_ENRICHED"],default="OPPOSITE_OR_UNSTABLE")
x["display_rank"]=np.minimum(x.EarlyHigh_FDR.rank(method="first"),x.LateHigh_FDR.rank(method="first"))
x.sort_values(["display_rank","GO_ID"]).to_csv(COMP/"EarlyHigh_vs_LateHigh_GO_comparison.csv",index=False)

# Primary-vs-promoter/gene-body-only mapping sensitivity.
maps={"EarlyHigh":pd.read_csv(MAP/"EarlyHigh_peak_to_gene.csv"),"LateHigh":pd.read_csv(MAP/"LateHigh_peak_to_gene.csv")}
srows=[]
for m in ("EarlyHigh","LateHigh"):
 mp=maps[m]; pg=set(mp.loc[mp.gene_id.fillna("").ne(""),"gene_symbol"]); sg=set(mp.loc[mp.annotation.isin(["promoter","5' UTR","exon","intron"]),"gene_symbol"])
 p=full[m]; q=strict[m]; z=p.merge(q,on="GO_ID",suffixes=("_primary","_strict"))
 rho=float(spearmanr(z.log2_corrected_odds_ratio_primary,z.log2_corrected_odds_ratio_strict).statistic)
 direction=float((np.sign(z.log2_corrected_odds_ratio_primary)==np.sign(z.log2_corrected_odds_ratio_strict)).mean())
 pt=set(p.sort_values(["BH_adjusted_pvalue","GO_ID"]).head(10).GO_ID); st=set(q.sort_values(["BH_adjusted_pvalue","GO_ID"]).head(10).GO_ID)
 srows.append({"module":m,"primary_gene_count":len(pg),"strict_gene_count":len(sg),"gene_set_overlap_fraction":len(pg&sg)/max(1,len(pg)),
  "common_term_count":len(z),"effect_spearman":rho,"top10_overlap_fraction":len(pt&st)/10,"direction_concordance":direction})
s=pd.DataFrame(srows)
high=(s.gene_set_overlap_fraction.ge(.7)&s.effect_spearman.ge(.8)&s.top10_overlap_fraction.ge(.5)&s.direction_concordance.ge(.8)).all()
moderate=(s.gene_set_overlap_fraction.ge(.5)&s.effect_spearman.ge(.6)&s.top10_overlap_fraction.ge(.25)&s.direction_concordance.ge(.6)).all()
sens="HIGH_CONCORDANCE" if high else ("MODERATE_CONCORDANCE" if moderate else "LOW_CONCORDANCE")
s["mapping_sensitivity_status"]=sens; s.to_csv(SENS/"GO_MAPPING_SENSITIVITY.csv",index=False)
(SENS/"GO_MAPPING_SENSITIVITY.md").write_text("# GO mapping sensitivity\n\n"
 f"Status: **{sens}**\n\nThe certified nearest-gene ≤100 kb analysis was compared with the stricter promoter/5′UTR/exon/intron-only mapping.\n\n"+
 "\n".join(f"- {z.module}: gene retention={z.gene_set_overlap_fraction:.3f}, effect Spearman={z.effect_spearman:.3f}, top-10 overlap={z.top10_overlap_fraction:.3f}, direction concordance={z.direction_concordance:.3f}" for z in s.itertuples())+"\n")

universe=json.loads((META/"GO_resource_and_universe_summary.json").read_text()); quality=[]
for m in ("EarlyHigh","LateHigh"):
 n=nr[m]; fg=universe["primary_foreground_GO_eligible"][m]; finite=np.isfinite(full[m][["corrected_odds_ratio","log2_corrected_odds_ratio","finite_adjusted_pvalue","minus_log10_FDR"]]).all().all()
 support=int(((n.foreground_gene_count>=3)&(n.log2_corrected_odds_ratio>=.5)).sum()) if len(n) else 0
 strong=len(n)>=5 and fg>=10 and support>=3 and sens=="HIGH_CONCORDANCE" and finite
 moderate=len(n)>=2 and fg>=5 and int(((n.foreground_gene_count>=3)&(n.log2_corrected_odds_ratio>=.25)).sum())>=1 and sens!="LOW_CONCORDANCE" and finite
 status="STRONG_GO_ENRICHMENT" if strong else ("MODERATE_GO_ENRICHMENT" if moderate else ("INSUFFICIENT_ENRICHMENT" if len(n)==0 else "WEAK_OR_MAPPING_SENSITIVE"))
 quality.append({"module":m,"GO_eligible_foreground_gene_count":fg,"nonredundant_enriched_GO_count":len(n),"supported_effect_count":support,
  "mapping_sensitivity_status":sens,"all_finite":bool(finite),"quality_status":status})
pd.DataFrame(quality).to_csv(META/"PHASE6_GO_QUALITY_RESULTS.csv",index=False)
summary={"mapping_sensitivity_status":sens,"quality":quality,"shared_GO_count":int((x.category=="SHARED_ENRICHED").sum()),
 "early_specific_GO_count":int((x.category=="EARLY_SPECIFIC").sum()),"late_specific_GO_count":int((x.category=="LATE_SPECIFIC").sum())}
(META/"phase6_analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

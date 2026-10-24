#!/usr/bin/env python3
"""Certified GO BP ORA for primary and stricter mapping sensitivity."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
from collections import defaultdict
import gzip, json, math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/GO_phase6_v1"
MAP=OUT/"gene_mapping"; META=OUT/"metadata"; SENS=OUT/"sensitivity"
OBO=SINGLE_DOWNSTREAM_ROOT/"resources/gene_sets/go_2026_06_19/go-basic.obo"
GAF=SINGLE_DOWNSTREAM_ROOT/"resources/gene_sets/go_2026_06_19/MOUSE-mod.gaf.gz"

terms={}; parents=defaultdict(set); cur={}
def finish():
 if cur.get("id") and cur.get("namespace")=="biological_process" and not cur.get("obsolete"):
  terms[cur["id"]]=cur.get("name",cur["id"]); parents[cur["id"]].update(cur.get("parents",set()))
for line in OBO.read_text().splitlines():
 if line=="[Term]": finish(); cur={"parents":set()}
 elif line.startswith("["): finish(); cur={}
 elif line.startswith("id: GO:"): cur["id"]=line[4:]
 elif line.startswith("name: "): cur["name"]=line[6:]
 elif line.startswith("namespace: "): cur["namespace"]=line[11:]
 elif line.startswith("is_a: "): cur.setdefault("parents",set()).add(line.split()[1])
 elif line.startswith("relationship: part_of "): cur.setdefault("parents",set()).add(line.split()[2])
 elif line=="is_obsolete: true": cur["obsolete"]=True
finish(); parents={t:{p for p in ps if p in terms} for t,ps in parents.items() if t in terms}
direct=defaultdict(set); alias={}
with gzip.open(GAF,"rt") as f:
 for line in f:
  if line.startswith("!"): continue
  z=line.rstrip().split("\t")
  if len(z)<15 or z[8]!="P" or "NOT" in z[3].split("|") or z[4] not in terms: continue
  sym=z[2]; direct[sym].add(z[4]); alias[sym.upper()]=sym
  for a in z[10].split("|"):
   if a: alias.setdefault(a.upper(),sym)
memo={}
def ancestors(t):
 if t in memo:return memo[t]
 ans={t}
 for p in parents.get(t,()): ans.update(ancestors(p))
 memo[t]=ans; return ans
gene_to_go={g:set().union(*(ancestors(t) for t in ts)) for g,ts in direct.items()}
def canonical(s): return alias.get(str(s).upper(),str(s))

tabs={"EarlyHigh":pd.read_csv(MAP/"EarlyHigh_peak_to_gene.csv"),"LateHigh":pd.read_csv(MAP/"LateHigh_peak_to_gene.csv"),
      "Background":pd.read_csv(MAP/"tested_background_peak_to_gene.csv")}
for x in tabs.values(): x["GO_symbol"]=x.gene_symbol.fillna("").map(canonical)
def gene_sets(strict):
 def genes(x):
  keep=x.gene_id.fillna("").ne("")
  if strict: keep &= x.annotation.isin(["promoter","5' UTR","exon","intron"])
  return set(x.loc[keep,"GO_symbol"])&set(gene_to_go)
 return genes(tabs["Background"]),genes(tabs["EarlyHigh"]),genes(tabs["LateHigh"])
def enrich(module,fg,bg):
 termbg=defaultdict(set)
 for g in bg:
  for t in gene_to_go[g]: termbg[t].add(g)
 rows=[]; n=len(fg); N=len(bg)
 for t,bgenes in termbg.items():
  ov=fg&bgenes
  if not ov: continue
  a=len(ov); b=n-a; c=len(bgenes)-a; d=N-a-b-c
  raw_or,p=fisher_exact([[a,b],[c,d]],alternative="greater")
  cor=((a+.5)*(d+.5))/((b+.5)*(c+.5)); rows.append({"module":module,"GO_ID":t,"term":terms[t],
   "foreground_gene_count":a,"foreground_gene_total":n,"background_gene_count":len(bgenes),"background_gene_total":N,
   "gene_ratio":a/n,"background_ratio":len(bgenes)/N,"raw_odds_ratio":raw_or,"corrected_odds_ratio":cor,
   "log2_corrected_odds_ratio":math.log2(cor),"raw_pvalue":p,"genes":";".join(sorted(ov))})
 r=pd.DataFrame(rows); r["BH_adjusted_pvalue"]=multipletests(r.raw_pvalue,method="fdr_bh")[1]
 r["finite_adjusted_pvalue"]=r.BH_adjusted_pvalue.clip(lower=1e-300); r["minus_log10_FDR"]=np.minimum(-np.log10(r.finite_adjusted_pvalue),300)
 r["enriched"]=(r.BH_adjusted_pvalue<.05)&(r.corrected_odds_ratio>1)
 return r.sort_values(["BH_adjusted_pvalue","GO_ID"])

bg,ef,lf=gene_sets(False); sbg,sef,slf=gene_sets(True)
primary={}; strict={}
for m,fg,sfg in [("EarlyHigh",ef,sef),("LateHigh",lf,slf)]:
 r=enrich(m,fg,bg); sr=enrich(m,sfg,sbg); primary[m]=r; strict[m]=sr
 r.to_csv(OUT/m/"full_GO_BP_raw.csv",index=False)
 finite=r.copy(); finite["raw_odds_ratio"]=finite.raw_odds_ratio.map(lambda v:str(v) if np.isfinite(v) else "undefined_zero_denominator")
 finite.to_csv(OUT/m/"full_GO_BP_finite.csv",index=False)
 finite[finite.enriched].to_csv(OUT/m/"filtered_GO_BP.csv",index=False)
 sr.to_csv(SENS/f"{m}_promoter_gene_body_only_GO_BP.csv",index=False)
 (OUT/m/"metadata.json").write_text(json.dumps({"module":m,"mapping":"certified nearest-gene <=100kb",
   "foreground_GO_eligible_gene_count":len(fg),"background_GO_eligible_gene_count":len(bg),
   "GO_ontology_release":"2026-06-15","GO_product_release":"2026-06-19","not_GREAT":True,
   "test":"one-sided Fisher","multiple_testing":"BH","background_source":"genes mapped from 6540 frozen tested peaks"},indent=2)+"\n")
(META/"GO_resource_and_universe_summary.json").write_text(json.dumps({"ontology":str(OBO),"annotations":str(GAF),
 "primary_background_mapped_gene_count":tabs["Background"].loc[tabs["Background"].gene_id.fillna("").ne(""),"gene_id"].nunique(),
 "primary_background_GO_eligible_gene_count":len(bg),"strict_background_GO_eligible_gene_count":len(sbg),
 "primary_foreground_GO_eligible":{"EarlyHigh":len(ef),"LateHigh":len(lf)},
 "strict_foreground_GO_eligible":{"EarlyHigh":len(sef),"LateHigh":len(slf)}},indent=2)+"\n")
print(json.dumps({"background_GO_eligible":len(bg),"EarlyHigh_GO_eligible":len(ef),"LateHigh_GO_eligible":len(lf),
 "EarlyHigh_enriched":int(primary["EarlyHigh"].enriched.sum()),"LateHigh_enriched":int(primary["LateHigh"].enriched.sum())},indent=2))

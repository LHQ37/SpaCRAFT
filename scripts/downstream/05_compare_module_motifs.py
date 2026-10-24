#!/usr/bin/env python3
"""Deterministic family reduction, module comparison, and quality classification."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/motif_phase5_v1"
COMP=OUT/"comparison"; META=OUT/"metadata"; SENS=OUT/"sensitivity"
full={m:pd.read_csv(OUT/m/"full_motif_enrichment_finite.csv") for m in ("EarlyHigh","LateHigh")}
maps=[]
for m,x in full.items():
 x=x.sort_values(["BH_adjusted_pvalue","log2_corrected_odds_ratio","motif_id"],ascending=[True,False,True])
 rep=x.drop_duplicates("motif_family")
 chosen=set(rep.motif_id)
 for z in x.itertuples():
  maps.append({"module":m,"motif_family":z.motif_family,"motif_id":z.motif_id,
   "representative_motif_id":rep.set_index("motif_family").loc[z.motif_family,"motif_id"],
   "is_representative":z.motif_id in chosen})
pd.DataFrame(maps).to_csv(COMP/"nonredundant_family_mapping.csv",index=False)

cols=["motif_id","TF_symbol","motif_family","log2_corrected_odds_ratio","BH_adjusted_pvalue","enriched",
      "foreground_peak_count","foreground_total"]
a=full["EarlyHigh"][cols].rename(columns={c:f"EarlyHigh_{c}" for c in cols if c not in ("motif_id","TF_symbol","motif_family")})
b=full["LateHigh"][cols].rename(columns={c:f"LateHigh_{c}" for c in cols if c not in ("motif_id","TF_symbol","motif_family")})
x=a.merge(b,on=["motif_id","TF_symbol","motif_family"],validate="one_to_one")
x["EarlyHigh_log2OR"]=x.EarlyHigh_log2_corrected_odds_ratio; x["EarlyHigh_FDR"]=x.EarlyHigh_BH_adjusted_pvalue
x["LateHigh_log2OR"]=x.LateHigh_log2_corrected_odds_ratio; x["LateHigh_FDR"]=x.LateHigh_BH_adjusted_pvalue
x["enrichment_difference"]=x.EarlyHigh_log2OR-x.LateHigh_log2OR
ee=x.EarlyHigh_enriched.astype(bool); le=x.LateHigh_enriched.astype(bool); same=np.sign(x.EarlyHigh_log2OR)==np.sign(x.LateHigh_log2OR)
x["direction_category"]=np.select([ee&le,ee&~le,~ee&le,~ee&~le&same],
 ["SHARED_ENRICHED","EARLY_SPECIFIC","LATE_SPECIFIC","SHARED_NOT_ENRICHED"],default="OPPOSITE_OR_UNSTABLE")
rankE=x.EarlyHigh_FDR.rank(method="first"); rankL=x.LateHigh_FDR.rank(method="first")
x["display_rank"]=np.minimum(rankE,rankL)
x.sort_values(["display_rank","motif_id"]).to_csv(COMP/"EarlyHigh_vs_LateHigh_motif_comparison.csv",index=False)

sens=json.loads((META/"background_sensitivity_summary.json").read_text())["status"]
q=[]
for m in ("EarlyHigh","LateHigh"):
 nr=pd.read_csv(OUT/m/"nonredundant_motif_enrichment_finite.csv"); en=nr[nr.enriched.astype(bool)]
 finite=np.isfinite(nr[["corrected_odds_ratio","log2_corrected_odds_ratio","finite_adjusted_pvalue","minus_log10_FDR"]]).all().all()
 strong=len(en)>=5 and (en.log2_corrected_odds_ratio>=.5).sum()>=3 and sens=="HIGH_CONCORDANCE" and finite
 moderate=len(en)>=2 and (en.log2_corrected_odds_ratio>=.25).sum()>=1 and sens!="LOW_CONCORDANCE" and finite
 status="STRONG_MOTIF_ENRICHMENT" if strong else ("MODERATE_MOTIF_ENRICHMENT" if moderate else ("INSUFFICIENT_ENRICHMENT" if len(en)==0 else "WEAK_OR_BACKGROUND_SENSITIVE"))
 q.append({"module":m,"nonredundant_enriched_motif_count":len(en),"effect_ge_0_5_count":int((en.log2_corrected_odds_ratio>=.5).sum()),
  "background_sensitivity_status":sens,"all_finite":bool(finite),"quality_status":status})
pd.DataFrame(q).to_csv(META/"PHASE5_MOTIF_QUALITY_RESULTS.csv",index=False)
counts=x.direction_category.value_counts()
summary={"background_sensitivity_status":sens,"quality":q,
 "shared_enriched_motif_count":int(counts.get("SHARED_ENRICHED",0)),
 "early_specific_motif_count":int(counts.get("EARLY_SPECIFIC",0)),
 "late_specific_motif_count":int(counts.get("LATE_SPECIFIC",0))}
(META/"phase5_analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

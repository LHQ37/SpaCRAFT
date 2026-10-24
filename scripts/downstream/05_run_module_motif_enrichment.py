#!/usr/bin/env python3
"""Bias-controlled module motif enrichment plus unadjusted tested-universe sensitivity."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json, math, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact, spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/motif_phase5_v1"
MM=OUT/"motif_matrix"; SEQ=OUT/"sequence_audit"; SENS=OUT/"sensitivity"; META=OUT/"metadata"
X=sparse.load_npz(MM/"C5_tested_peak_motif_presence_sparse.npz").tocsc()
cov=pd.read_csv(SEQ/"C5_tested_peak_sequence_covariates.csv.gz"); motifs=pd.read_csv(MM/"motif_metadata.csv")
assert X.shape==(6540,len(motifs)) and len(motifs)==879
Z=np.c_[cov.GC_fraction,np.log(cov.peak_length),cov.mean_accessibility].astype(float)
Z=(Z-Z.mean(0))/Z.std(0); Z=sm.add_constant(Z)
all_results={}; sensrows=[]; summaries=[]
for module in ("EarlyHigh","LateHigh"):
 y=cov.module_membership.eq(module).astype(float).to_numpy(); assert y.sum()==(126 if module=="EarlyHigh" else 106)
 rows=[]; rawrows=[]
 for j,m in motifs.iterrows():
  x=X[:,j].toarray().ravel().astype(float); a=int(x[y==1].sum()); b=int(y.sum()-a); c=int(x[y==0].sum()); d=int((y==0).sum()-c)
  raw_or,fp=fisher_exact([[a,b],[c,d]],alternative="two-sided")
  cor=((a+.5)*(d+.5))/((b+.5)*(c+.5)); logor=math.log2(cor)
  design=np.c_[Z,x]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit=sm.GLM(y,design,family=sm.families.Binomial()).fit(maxiter=100,disp=0)
   beta=float(fit.params[-1]); lp=float(fit.pvalues[-1])
   if not np.isfinite(beta) or not np.isfinite(lp): raise ValueError("nonfinite")
  except Exception:
   beta=0.0; lp=1.0
  base={"motif_id":m.motif_id,"TF_symbol":m.TF_symbol,"motif_family":m.motif_family,
   "foreground_peak_count":a,"foreground_total":int(y.sum()),"background_peak_count":c,"background_total":int((y==0).sum()),
   "raw_odds_ratio":raw_or,"corrected_odds_ratio":cor,"log2_corrected_odds_ratio":logor,
   "raw_pvalue":lp,"unadjusted_fisher_pvalue":fp,"logistic_motif_coefficient":beta}
  rows.append(base); rawrows.append(base)
 r=pd.DataFrame(rows); r["BH_adjusted_pvalue"]=multipletests(r.raw_pvalue,method="fdr_bh")[1]
 r["finite_adjusted_pvalue"]=r.BH_adjusted_pvalue.clip(lower=1e-300)
 r["minus_log10_FDR"]=np.minimum(-np.log10(r.finite_adjusted_pvalue),300)
 r["unadjusted_BH_FDR"]=multipletests(r.unadjusted_fisher_pvalue,method="fdr_bh")[1]
 r["enriched"]=(r.BH_adjusted_pvalue<=.05)&(r.logistic_motif_coefficient>0)&(r.corrected_odds_ratio>1)
 raw=r.copy(); raw.to_csv(OUT/module/"full_motif_enrichment_raw.csv",index=False)
 finite=r.copy(); finite["raw_odds_ratio"]=finite.raw_odds_ratio.map(lambda v: str(v) if np.isfinite(v) else "undefined_zero_denominator")
 finite.to_csv(OUT/module/"full_motif_enrichment_finite.csv",index=False)
 nr=finite.sort_values(["BH_adjusted_pvalue","log2_corrected_odds_ratio","motif_id"],ascending=[True,False,True]).drop_duplicates("motif_family")
 nr.to_csv(OUT/module/"nonredundant_motif_enrichment_finite.csv",index=False)
 top=nr.sort_values(["enriched","BH_adjusted_pvalue","log2_corrected_odds_ratio","motif_id"],ascending=[False,True,False,True]).head(20)
 top.to_csv(OUT/module/"top_motifs.csv",index=False)
 meta={"module":module,"foreground_total":int(y.sum()),"background_total":int((y==0).sum()),"motif_count":len(r),
  "bias_control":"logistic membership ~ motif + standardized GC + standardized log length + standardized mean C5 accessibility",
  "multiple_testing":"BH across 879 motifs","finite_effect":"Haldane-Anscombe +0.5","enriched_definition":"BH FDR<=0.05, beta>0, corrected OR>1"}
 (OUT/module/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
 # Sensitivity metrics are computed over finite effects and ranks.
 rho=float(spearmanr(r.logistic_motif_coefficient,r.log2_corrected_odds_ratio).statistic)
 direction=float((np.sign(r.logistic_motif_coefficient)==np.sign(r.log2_corrected_odds_ratio)).mean())
 ptop=set(r.sort_values(["BH_adjusted_pvalue","logistic_motif_coefficient"],ascending=[True,False]).head(20).motif_id)
 utop=set(r.sort_values(["unadjusted_BH_FDR","log2_corrected_odds_ratio"],ascending=[True,False]).head(20).motif_id)
 overlap=len(ptop&utop)/20
 sensrows.append({"module":module,"effect_spearman":rho,"top20_overlap_fraction":overlap,"direction_concordance":direction})
 all_results[module]=r

s=pd.DataFrame(sensrows)
high=(s.effect_spearman.ge(.8)&s.top20_overlap_fraction.ge(.5)&s.direction_concordance.ge(.8)).all()
moderate=(s.effect_spearman.ge(.6)&s.top20_overlap_fraction.ge(.25)&s.direction_concordance.ge(.6)).all()
status="HIGH_CONCORDANCE" if high else ("MODERATE_CONCORDANCE" if moderate else "LOW_CONCORDANCE")
s["background_sensitivity_status"]=status; s.to_csv(SENS/"MOTIF_BACKGROUND_SENSITIVITY.csv",index=False)
(SENS/"MOTIF_BACKGROUND_SENSITIVITY.md").write_text(
 "# Motif background sensitivity\n\n"
 f"Status: **{status}**\n\nPrimary bias-controlled logistic coefficients were compared with unadjusted "
 "tested-universe Haldane–Anscombe log2 odds ratios. The unadjusted analysis is sensitivity-only.\n\n"+
 "\n".join(f"- {z.module}: effect Spearman={z.effect_spearman:.4f}, top-20 overlap={z.top20_overlap_fraction:.3f}, direction concordance={z.direction_concordance:.3f}" for z in s.itertuples())+"\n")
(META/"background_sensitivity_summary.json").write_text(json.dumps({"status":status,"modules":sensrows},indent=2)+"\n")
print(json.dumps({"background_sensitivity_status":status,"modules":sensrows},indent=2))

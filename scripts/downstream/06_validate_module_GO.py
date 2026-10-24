#!/usr/bin/env python3
"""Mechanical and policy validation for Phase 6."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json, re
import numpy as np
import pandas as pd
from PIL import Image

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/GO_phase6_v1"
P3=ROOT/"results/accessibility_phase3_v1/downstream_inputs"; MAP=OUT/"gene_mapping"; META=OUT/"metadata"
e=pd.read_csv(P3/"C5_EarlyHigh_peaks_for_gene_mapping.csv"); l=pd.read_csv(P3/"C5_LateHigh_peaks_for_gene_mapping.csv"); u=pd.read_csv(P3/"C5_tested_peak_background_for_motif.csv")
em=pd.read_csv(MAP/"EarlyHigh_peak_to_gene.csv"); lm=pd.read_csv(MAP/"LateHigh_peak_to_gene.csv"); bm=pd.read_csv(MAP/"tested_background_peak_to_gene.csv")
checks={}
checks["frozen_peak_counts"]=len(e)==len(em)==126 and len(l)==len(lm)==106 and len(u)==len(bm)==6540
checks["foregrounds_are_disjoint_subsets"]=set(e.peak_id)<=set(u.peak_id) and set(l.peak_id)<=set(u.peak_id) and not set(e.peak_id)&set(l.peak_id)
checks["mapping_exact_peak_order"]=e.peak_id.equals(em.peak_id) and l.peak_id.equals(lm.peak_id) and u.peak_id.equals(bm.peak_id)
checks["certified_mapping_rule"]=set(pd.concat([em,lm,bm]).mapping_rule)<=set(["DIRECT_PROMOTER_OR_GENE_BODY","NEAREST_GENE_WITHIN_100KB","UNMAPPED_BEYOND_100KB","UNMAPPED_NONSTANDARD_CHROMOSOME","UNMAPPED_PARSE"])
checks["background_from_tested_peaks"]=(OUT/"background/C5_tested_peak_background_gene_table.csv").exists()
analysis_scripts=[ROOT/"scripts"/n for n in ["06_map_module_peaks_to_genes.py","06_run_module_GO_enrichment.py","06_compare_module_GO.py","06_plot_module_GO.py"]]
checks["phase4_not_used"]=not any("gene_activity_phase4" in p.read_text(errors="ignore") for p in analysis_scripts)
finite=["corrected_odds_ratio","log2_corrected_odds_ratio","finite_adjusted_pvalue","minus_log10_FDR"]
checks["formal_values_finite"]=all(np.isfinite(pd.read_csv(OUT/m/"full_GO_BP_finite.csv")[finite]).all().all() for m in ("EarlyHigh","LateHigh"))
checks["mapping_sensitivity_complete"]=(OUT/"sensitivity/GO_MAPPING_SENSITIVITY.md").exists()
figs=list((ROOT/"figures/candidates/GO_phase6_v1").glob("*.png"))
checks["candidate_figures_readable"]=len(figs)==4 and all((lambda p:p.stat().st_size>1000 and Image.open(p).verify() is None)(p) for p in figs)
checks["not_GREAT"]=all(json.loads((OUT/m/"metadata.json").read_text())["not_GREAT"] for m in ("EarlyHigh","LateHigh"))
checks["final_figure_not_assembled"]=not (ROOT/"figures/final/Figure6.png").exists()
checks["module_membership_unchanged"]=set(em.peak_id)==set(e.peak_id) and set(lm.peak_id)==set(l.peak_id)
report={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
 "policy":{"formal_replicate_inference_performed":False,"spot_as_replicate_used":False,"models_retrained":False,
 "embedding_recomputed":False,"clusters_recomputed":False,"formal_inputs_modified":False,
 "GREAT_run":False,"gene_activity_run":False,"motif_run":False,"final_figure_assembled":False}}
(META/"phase6_validation_report.json").write_text(json.dumps(report,indent=2,default=lambda x:bool(x))+"\n")
assert report["status"]=="PASS",report
print(json.dumps(report,indent=2,default=lambda x:bool(x)))

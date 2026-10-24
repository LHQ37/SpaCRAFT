#!/usr/bin/env python3
"""Independent objective validation of Phase 1 composition outputs."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT=MULTI_DOWNSTREAM_ROOT
RES=ROOT/"results/composition_v1"; FIG=ROOT/"figures/candidates/composition_v1"
checks={}
for ds,n,k,ns in [("six_slice",17787,11,6),("four_slice",7118,16,4)]:
    c=pd.read_csv(RES/ds/"cluster_counts_by_slice.csv")
    p=pd.read_csv(RES/ds/"cluster_proportions_by_slice.csv")
    checks[f"{ds}_spot_total"] = int(c.groupby(["slice","cluster"]).spot_count.first().sum()) == n
    checks[f"{ds}_cluster_range"] = sorted(c.cluster.unique().tolist()) == list(range(k))
    checks[f"{ds}_slice_count"] = c.slice.nunique() == ns
    checks[f"{ds}_proportion_sums"] = bool(((p.groupby("slice").cluster_proportion_within_slice.sum()-1).abs()<=1e-10).all())
    checks[f"{ds}_finite"] = bool(np.isfinite(p.select_dtypes("number")).all().all())
checks["six_stage_two_slices"] = bool((pd.read_csv(
    RES/"six_slice/cluster_proportions_by_stage_descriptive.csv").groupby("stage").n_slices.first()==2).all())
checks["coverage_gate"] = all((RES/d/"cluster_downstream_eligibility.csv").is_file() for d in ("six_slice","four_slice"))
pngs=sorted(FIG.glob("*.png"))
checks["candidate_png_count_9"] = len(pngs)==9
checks["candidate_png_readable"] = True
for p in pngs:
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception:
        checks["candidate_png_readable"]=False
checks["no_inferential_columns"] = not any(
    x.lower() in {"pvalue","p_value","p-value","fdr","qvalue","significance"}
    for f in RES.rglob("*.csv") for x in pd.read_csv(f,nrows=1).columns)
checks["formal_inference_performed"] = False
checks["spot_as_replicate_used"] = False
status="COMPLETE" if all(v is True for k,v in checks.items()
                         if k not in ("formal_inference_performed","spot_as_replicate_used")) else "FAILED"
report={"phase1_composition_status":status,"checks":checks}
(RES/"metadata/validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
if status!="COMPLETE":
    raise SystemExit(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

#!/usr/bin/env bash
set -euo pipefail
project="${SPACRAFT_SINGLE_DOWNSTREAM_ROOT:-outputs/downstream/single_slice}"
launcher="$project/environment/run_in_single_downstream.sh"
cd "$project"
{
  printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
  printf 'command=bash %s/scripts/run_resume_after_marker_peaks.sh\n' "$project"
  printf 'seed=1234\n'
} > audit/RESUME_COMMAND_HISTORY.txt
{
  uname -a
  bash "$launcher" python --version
  bash "$launcher" python -c 'import anndata, numpy, pandas, scipy, matplotlib, snapatac2; print("anndata="+anndata.__version__); print("numpy="+numpy.__version__); print("pandas="+pandas.__version__); print("scipy="+scipy.__version__); print("matplotlib="+matplotlib.__version__); print("snapatac2="+snapatac2.__version__)'
} > audit/RESUME_SOFTWARE_VERSIONS.txt 2>&1
cat > environment/CHROMVAR_ENVIRONMENT_COMMANDS.md <<'EOF'
# Isolated chromVAR environment

Status: **ENVIRONMENT_UNAVAILABLE**

Audit found cached R bases but no cached/validated chromVAR, motifmatchr, TFBSTools,
JASPAR data, BSgenome mm10, or the complete Bioconductor dependency closure.
No existing environment was modified and no ordinary motif count was substituted.

When network installation is authorized, create the independent environment only:

```bash
conda create -n single_downstream_r -c conda-forge -c bioconda r-base=4.4 bioconductor-chromvar bioconductor-motifmatchr bioconductor-tfbstools bioconductor-jaspar2024 bioconductor-genomicranges bioconductor-summarizedexperiment bioconductor-bsgenome.mmusculus.ucsc.mm10 r-matrix r-data.table
conda run -n single_downstream_r Rscript -e 'library(chromVAR); library(motifmatchr); library(TFBSTools); library(JASPAR2024); library(GenomicRanges); library(SummarizedExperiment); library(BSgenome.Mmusculus.UCSC.mm10); library(Matrix); library(data.table); sessionInfo()'
```
EOF
bash "$launcher" python scripts/09_resume_after_marker_peaks.py
printf 'end_time=%s\n' "$(date --iso-8601=seconds)" >> audit/RESUME_COMMAND_HISTORY.txt

#!/usr/bin/env bash
set -euo pipefail
root="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
py="${PYTHON:-python}"
mkdir -p "$root/logs"
{
  "$py" "$root/scripts/04_build_gene_activity_matrix.py"
  "$py" "$root/scripts/04_analyze_gene_activity_dynamics.py"
  "$py" "$root/scripts/04_analyze_module_gene_concordance.py"
  "$py" "$root/scripts/04_plot_gene_activity_dynamics.py"
  "$py" "$root/scripts/04_validate_gene_activity.py"
} 2>&1 | tee "$root/logs/phase4_gene_activity.log"

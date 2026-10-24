#!/usr/bin/env bash
set -euo pipefail
root="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
py="${PYTHON:-python}"
mkdir -p "$root/logs"
{
 "$py" "$root/scripts/07_compute_spatial_module_scores.py"
 "$py" "$root/scripts/07_analyze_module_score_QC.py"
 "$py" "$root/scripts/07_plot_spatial_module_scores.py"
 "$py" "$root/scripts/07_audit_four_slice_projection.py"
 "$py" "$root/scripts/07_validate_spatial_module_scores.py"
} 2>&1 | tee "$root/logs/phase7_module_spatial_scores.log"

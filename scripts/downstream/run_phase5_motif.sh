#!/usr/bin/env bash
set -euo pipefail
root="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
py="${PYTHON:-python}"
mkdir -p "$root/logs"
{
 "$py" "$root/scripts/05_scan_module_motifs.py"
 "$py" "$root/scripts/05_run_module_motif_enrichment.py"
 "$py" "$root/scripts/05_compare_module_motifs.py"
 "$py" "$root/scripts/05_plot_module_motifs.py"
 "$py" "$root/scripts/05_validate_module_motifs.py"
} 2>&1 | tee "$root/logs/phase5_motif_enrichment.log"

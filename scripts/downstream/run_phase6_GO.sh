#!/usr/bin/env bash
set -euo pipefail
root="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
py="${PYTHON:-python}"
mkdir -p "$root/logs"
{
 "$py" "$root/scripts/06_map_module_peaks_to_genes.py"
 "$py" "$root/scripts/06_run_module_GO_enrichment.py"
 "$py" "$root/scripts/06_compare_module_GO.py"
 "$py" "$root/scripts/06_plot_module_GO.py"
 "$py" "$root/scripts/06_validate_module_GO.py"
} 2>&1 | tee "$root/logs/phase6_GO_enrichment.log"

#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
PY="${PYTHON:-python}"
"$PY" "$ROOT/scripts/02_build_cluster_slice_pseudobulk.py"
"$PY" "$ROOT/scripts/02_analyze_stage_associated_peaks.py"
"$PY" "$ROOT/scripts/02_plot_stage_associated_peaks.py"
"$PY" "$ROOT/scripts/02_validate_stage_associated_peaks.py"

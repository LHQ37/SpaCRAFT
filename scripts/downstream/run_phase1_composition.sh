#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
PY="${PYTHON:-python}"
"$PY" "$ROOT/scripts/01_run_cluster_composition.py"
"$PY" "$ROOT/scripts/01_plot_cluster_composition.py"
"$PY" "$ROOT/scripts/01_validate_cluster_composition.py"

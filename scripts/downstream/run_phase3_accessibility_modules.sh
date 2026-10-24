#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPACRAFT_MULTI_DOWNSTREAM_ROOT:-outputs/downstream/multi_slice}"
PY="${PYTHON:-python}"
"$PY" "$ROOT/scripts/03_build_accessibility_modules.py"
"$PY" "$ROOT/scripts/03_analyze_module_stability.py"
"$PY" "$ROOT/scripts/03_plot_accessibility_modules.py"
"$PY" "$ROOT/scripts/03_validate_accessibility_modules.py"

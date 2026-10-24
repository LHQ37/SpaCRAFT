#!/usr/bin/env bash
set -euo pipefail

project="${SPACRAFT_SINGLE_DOWNSTREAM_ROOT:-outputs/downstream/single_slice}"
launcher="$project/environment/run_in_single_downstream.sh"

cd "$project"

for protected in \
  results/by_slice/*/marker_peaks \
  results/by_slice/*/peak_annotation \
  results/by_slice/*/gene_activity
do
  test -d "$protected"
done

test -s resources/gene_sets/go_2026_06_19/go-basic.obo
test -s resources/gene_sets/go_2026_06_19/MOUSE-mod.gaf.gz
gzip -t resources/gene_sets/go_2026_06_19/MOUSE-mod.gaf.gz

bash "$launcher" python scripts/10_core_correction.py

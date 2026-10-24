"""Repository-local and user-configurable paths for executable workflows."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def configured_path(environment_variable, repository_default):
    value = os.environ.get(environment_variable)
    return Path(value).expanduser() if value else REPO_ROOT / repository_default


DATA_ROOT = configured_path("SPACRAFT_DATA_ROOT", "data/external")
GENOME_ROOT = configured_path("SPACRAFT_GENOME_ROOT", "resources/mm10")
MULTISLICE_RESULTS_ROOT = configured_path(
    "SPACRAFT_MULTISLICE_RESULTS_ROOT", "results/multi_slice")
MULTI_DOWNSTREAM_ROOT = configured_path(
    "SPACRAFT_MULTI_DOWNSTREAM_ROOT", "outputs/downstream/multi_slice")
SINGLE_DOWNSTREAM_ROOT = configured_path(
    "SPACRAFT_SINGLE_DOWNSTREAM_ROOT", "outputs/downstream/single_slice")
ABLATION_ROOT = configured_path(
    "SPACRAFT_ABLATION_ROOT", "outputs/ablation")
FIGURE_OUTPUT_ROOT = configured_path(
    "SPACRAFT_FIGURE_OUTPUT_ROOT", "outputs/figures")

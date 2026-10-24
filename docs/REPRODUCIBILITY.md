# Reproducibility

## Environment

```bash
conda env create -f environment.yml
conda activate spacraft
pip install -e .
```

The performance-summary component also requires the R package
`funkyheatmap`.

Training requires CUDA. Always choose a new `--output-root`; committed results
and release figures are read-only reference artifacts.

## Formal model entry points

Single-slice:

```bash
python scripts/single_slice/run_spacraft.py \
  --dataset E13_5-S1_atac \
  --input-root /path/to/single_slice_h5ad \
  --output-root outputs/single_slice
```

The formal seed is 1236. The model uses raw 19-dimensional LSI input, a hybrid
feature/spatial graph, a 19→50→10 encoder, adversarial reconstruction, MSE
reconstruction and graph-neighborhood InfoNCE. BANKSY is not used.

Multi-slice:

```bash
python scripts/multi_slice/run_spacraft.py \
  --dataset six_slices \
  --input-root /path/to/prepared/six_slices \
  --output-root outputs/multi_slice
```

The formal seed is 1234. Peak matrices are aligned before a joint TF-IDF/LSI
fit; 30 components are computed and the first is removed. BANKSY feature
augmentation is part of the formal SpaCRAFT multi-slice model (lambda 0.15,
`kgeom=10`, AGF multiplier 2.0, `mu=1.0`), followed by a within-slice hybrid
graph and a 29→50→10 encoder. Exact records are under `configs/`.

## Figure reproduction

Single-slice benchmark components:

```bash
python figures/single_slice/benchmark/plot_ground_truth.py
Rscript figures/single_slice/benchmark/plot_performance_summary.R
python figures/single_slice/benchmark/plot_spatial_comparison.py
python figures/single_slice/benchmark/plot_umap_comparison.py
```

The four outputs are assembled manually in Adobe Illustrator.

Automated manuscript figures:

```bash
python figures/multi_slice/plot_multislice_benchmark.py
python figures/downstream/plot_single_slice_downstream.py
python figures/downstream/plot_multislice_downstream.py
python figures/ablation/plot_ablation.py
```

The multi-slice benchmark inputs are committed under `results/multi_slice/`.
Set `SPACRAFT_MULTISLICE_COMPARISON_ROOT` only to use an equivalent external
frozen-results root. The single-slice downstream figure additionally needs
mm10 GENCODE vM25 annotation and JASPAR 2024 motifs. Place them below
`resources/mm10/` or set `SPACRAFT_GENOME_ROOT` to another resource root.

## Evaluation and ablation

The formal multi-slice method order is INSTINCT, PeakVI, scDREAMER, SEDR,
STAligner, GraphST and SpaCRAFT. INSTINCT is an external comparison method, not
an alias for SpaCRAFT. The seven metrics and balanced ranking tables are under
`results/multi_slice/evaluation/`.

The manuscript ablation contains only single-slice graph ablation and
single-slice loss ablation averaged over six datasets. The retained data
preparation entry point is
`scripts/ablation/run_single_slice_loss_ablation.py`; the figure reads the
committed summary tables and does not train.

## Script classification

- `scripts/{single_slice,multi_slice}/run_spacraft.py`: `FORMAL_ENTRYPOINT`.
- `figures/**/plot_*`: `FORMAL_ENTRYPOINT`.
- `scripts/ablation/run_single_slice_loss_ablation.py`: `DATA_PREPARATION`.
- `scripts/downstream/[00-10]_*.py`: `DATA_PREPARATION` or
  `FORMAL_HELPER` for the ordered downstream workflow.
- `scripts/downstream/run_*.sh`: `DATA_PREPARATION` launchers.
- `src/spacraft/`: importable package implementation (`FORMAL_HELPER`).

There are no supplementary or external-wrapper executables in this release;
the retained scripts are the formal model, evaluation, downstream and figure
entry points listed above.

## Data and external resources

Large H5AD inputs are intentionally ignored by Git; see `data/README.md`.
Third-party methods must be installed from their official distributions. Only
the frozen, lightweight result triplets needed by the figures are committed;
see `results/README.md` and `external/README.md`.

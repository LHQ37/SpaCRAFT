# SpaCRAFT

SpaCRAFT learns spatial representations from single- and multi-slice chromatin
accessibility data. The multi-slice model aligns peaks, fits joint TF-IDF/LSI,
augments features with BANKSY, builds within-slice hybrid graphs and trains a
graph-attention representation. The single-slice model uses the raw LSI path
and does not use BANKSY.

## Installation

```bash
conda env create -f environment.yml
conda activate spacraft
pip install -e .
spacraft --help
```

## Data

Large H5AD files are not distributed or tracked. Put the six exact
single-slice inputs in `data/` or pass another directory with `--input-root`.
The expected filenames, label columns, preprocessing and current provenance
limitation are documented in [data/README.md](data/README.md). Multi-slice
prepared inputs are supplied through `--input-root`.

## Formal model entry points

Single-slice:

```bash
python scripts/single_slice/run_spacraft.py \
  --dataset E13_5-S1_atac \
  --input-root /path/to/single_slice_h5ad \
  --output-root outputs/single_slice
```

Multi-slice:

```bash
python scripts/multi_slice/run_spacraft.py \
  --dataset six_slices \
  --input-root /path/to/prepared/six_slices \
  --output-root outputs/multi_slice
```

The fixed parameter records are under `configs/single_slice/` and
`configs/multi_slice/`. New runs should always use a new output directory.

## Evaluation

The formal multi-slice comparison order is INSTINCT, PeakVI, scDREAMER, SEDR,
STAligner, GraphST and SpaCRAFT. These six named comparison methods are external
baselines. Frozen figure inputs are stored under `results/`; third-party source
trees are not vendored.

## Manuscript figures

Automated figure entry points:

```bash
python figures/multi_slice/plot_multislice_benchmark.py
python figures/downstream/plot_single_slice_downstream.py
python figures/downstream/plot_multislice_downstream.py
python figures/ablation/plot_ablation.py
```

The single-slice benchmark uses four separately generated components:

```bash
python figures/single_slice/benchmark/plot_ground_truth.py
Rscript figures/single_slice/benchmark/plot_performance_summary.R
python figures/single_slice/benchmark/plot_spatial_comparison.py
python figures/single_slice/benchmark/plot_umap_comparison.py
```

They are assembled manually in Adobe Illustrator; there is no automated
assembly script. The exact mapping from code to release files is in
[docs/FIGURE_CODE_MAP.md](docs/FIGURE_CODE_MAP.md).

The single-slice downstream figure may use an equivalent external frozen
multi-slice result root via `SPACRAFT_MULTISLICE_COMPARISON_ROOT`. Its motif and
gene-activity panels need mm10 GENCODE vM25/JASPAR resources below
`resources/mm10/` or at `SPACRAFT_GENOME_ROOT`.

## Ablation and downstream preparation

The manuscript ablation contains only single-slice graph ablation and the
six-dataset-average single-slice loss ablation. Its preparation runner is
`scripts/ablation/run_single_slice_loss_ablation.py`. Ordered downstream data
preparation and validation helpers are under `scripts/downstream/`; the formal
plotting entry points read committed precomputed results and do not retrain.

## Repository layout

| Path | Purpose |
|---|---|
| `src/spacraft/` | SpaCRAFT package |
| `configs/` | fixed model and analysis settings |
| `data/` | local, ignored large inputs and data instructions |
| `scripts/` | formal training and data-preparation entry points |
| `figures/` | stable plotting entry points and release PNG/PDF files |
| `results/` | frozen lightweight figure inputs and summary tables |
| `external/` | third-party installation policy |
| `tests/` | release invariants |
| `docs/` | reproducibility and figure mapping |

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for fixed parameters,
resource requirements and executable classification.

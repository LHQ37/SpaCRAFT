# Figure code map

Only current manuscript figures and their stable entry points are listed here.

| Figure | Formal entry point | Repository input | Release output | Structure |
|---|---|---|---|---|
| Single-slice benchmark a | `figures/single_slice/benchmark/plot_ground_truth.py` | `data/E13_5-S1_atac.h5ad` | `ground_truth.{png,pdf}` | E13.5 ground truth |
| Single-slice benchmark b | `figures/single_slice/benchmark/plot_performance_summary.R` | `performance_summary_input.csv` | `performance_summary.{png,pdf}` | performance/ranking summary |
| Single-slice benchmark c | `figures/single_slice/benchmark/plot_spatial_comparison.py` | `results/single_slice/benchmark/oracle_cluster_labels/` | `spatial_comparison.{png,pdf}` | E13.5, eight methods |
| Single-slice benchmark d | `figures/single_slice/benchmark/plot_umap_comparison.py` | `results/single_slice/benchmark/oracle_umap_coordinates/` | `umap_comparison.{png,pdf}` | E13.5, eight methods; arrows added in Adobe Illustrator |
| Multi-slice benchmark | `figures/multi_slice/plot_multislice_benchmark.py` | `results/multi_slice/` | `figures/main/fig3_multi_slice_benchmark.{png,pdf}` | three panels: a metrics; b spatial; c UMAP |
| Single-slice downstream | `figures/downstream/plot_single_slice_downstream.py` | committed single-slice results plus mm10 resources | `Figure_single_slice_downstream.{png,pdf}` | a–f; Functional enrichment is f |
| Multi-slice downstream | `figures/downstream/plot_multislice_downstream.py` | `results/downstream/multi_slice/`, canonical observations and SpaCRAFT labels | `Figure_multislice_downstream.{png,pdf}` | a–g |
| Ablation | `figures/ablation/plot_ablation.py` | `results/ablation/*mean*.csv` | `Figure_ablation.{png,pdf}` | a graph ablation; b six-dataset loss average, y=0.4–0.7 |

The final single-slice benchmark is assembled manually in Adobe Illustrator
from its four component outputs. There is intentionally no Python assembly
script.

The SpaCRAFT architecture/framework figure source and release output are not
yet present. This remains a release TODO and is not replaced by an older model
diagram.

# Reference results

This directory contains only committed inputs needed to reproduce manuscript
figures or lightweight summary tables needed to verify reported results.

- `multi_slice/`: canonical observations, frozen method result triplets and
  evaluation tables used by the multi-slice figures.
- `single_slice/benchmark/`: frozen cluster labels and UMAP coordinates used by
  the single-slice benchmark components.
- `single_slice/spacraft/results/`: SpaCRAFT outputs used by the single-slice
  downstream figure.
- `downstream/multi_slice/`: precomputed inputs for the seven-panel multi-slice
  downstream figure.
- `ablation/`: lightweight final summary tables.

INSTINCT, PeakVI, scDREAMER, SEDR, STAligner and GraphST are the external
comparison methods. Their frozen result tables are preserved here; third-party
source code is not vendored.
The repository contains result tables only, not third-party source code.

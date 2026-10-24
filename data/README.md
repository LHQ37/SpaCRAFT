# Data

The working directory currently contains six local H5AD inputs used for the
single-slice experiments:

| Dataset identifier | Expected local file | Label column |
|---|---|---|
| E11.0 embryo | `E11_0-S1_atac.h5ad` | `Annotation_for_Combined` |
| E13.5 embryo | `E13_5-S1_atac.h5ad` | `Annotation_for_Combined` |
| E15.5 embryo | `E15_5-S1_atac.h5ad` | `Annotation_for_Combined` |
| E18.5 embryo | `E18_5-S1_atac.h5ad` | `Annotation_for_Combined` |
| normalized peak dataset | `adata_peaks_normalized.h5ad` | `ATAC_clusters` |
| S1 ATAC dataset | `s1_adata_atac_with_meta.h5ad` | `ATAC_clusters` |

These files are excluded by `.gitignore` and must not be uploaded. Their
embedded metadata does not record a reliable source paper, accession and
download URL. Those provenance fields must be supplied and verified by the
data owner before public release or redistribution; this repository does not
guess them from filenames. Until then, obtain the exact inputs from the study
authors and place them at the paths above.

The formal preprocessing is implemented by
`scripts/single_slice/run_spacraft.py`: peak alignment, cell/feature filtering,
fragment-count transform where configured, TF-IDF, 20-component truncated SVD
with the first component removed, and the dataset-specific label column listed
above. Exact parameters are in
`configs/single_slice/frozen_spacraft_single_slice.json`.

Multi-slice data are not distributed. A prepared six-slice directory must
contain `prepared_data/prepared_obs.csv`, `joint_lsi_raw.npy`, and
`hybrid_graph_list.npz`; pass its parent through `--input-root`.

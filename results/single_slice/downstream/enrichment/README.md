# Single-slice functional enrichment audit

The audit reuses the frozen E15.5 per-cluster input genes and BP universe from the current downstream workflow. BP, CC and MF use the same symbols and background; KEGG uses the mapped subset after explicit mouse SYMBOL-to-Entrez conversion. All categories use one-sided Fisher tests and BH correction independently within each cluster/category; significance is FDR < 0.05. No figure, model, cluster assignment, marker peak, or peak-to-gene result was changed.

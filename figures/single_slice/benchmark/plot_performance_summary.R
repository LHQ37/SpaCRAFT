#!/usr/bin/env Rscript

# Values come only from the frozen formal benchmark table
# performance_summary_input.csv.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(funkyheatmap)
  library(RColorBrewer)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}

INPUT <- file.path(script_dir, "performance_summary_input.csv")
OUT_PDF <- file.path(script_dir, "performance_summary.pdf")
OUT_PNG <- file.path(script_dir, "performance_summary.png")

METHOD_ORDER <- c(
  "SpaCRAFT", "INSTINCT", "SnapATAC2", "Descart",
  "epiScanpy", "PeakVI", "SMOPCA", "SCALE"
)
DATASET_ORDER <- c(
  "s1_adata_atac_with_meta", "E11_0-S1_atac", "E13_5-S1_atac",
  "E15_5-S1_atac", "E18_5-S1_atac", "adata_peaks_normalized"
)
DATASET_LABEL <- c(
  s1_adata_atac_with_meta = "P22",
  `E11_0-S1_atac` = "E11",
  `E13_5-S1_atac` = "E13",
  `E15_5-S1_atac` = "E15",
  `E18_5-S1_atac` = "E18",
  adata_peaks_normalized = "Emb"
)
METRICS <- c(
  "ARI", "NMI", "AMI", "FMI", "Homogeneity", "Completeness",
  "V-measure", "Overall"
)
METRIC_AVERAGE_METRICS <- c("NMI", "ARI", "AMI", "Overall")

raw <- read.csv(INPUT, check.names = FALSE) %>%
  rename(Method = method) %>%
  filter(Method %in% METHOD_ORDER, dataset %in% DATASET_ORDER) %>%
  select(Method, dataset, all_of(METRICS))

stopifnot(nrow(raw) == 48L)
stopifnot(!anyDuplicated(raw[c("Method", "dataset")]))
stopifnot(all(is.finite(as.matrix(raw[METRICS]))))

# Rank each method within every dataset (1 is best), then
# display its mean rank across the six datasets for each metric.
rank_data <- raw %>%
  group_by(dataset) %>%
  mutate(across(all_of(METRICS), ~ rank(-.x, ties.method = "min"),
                .names = "RANK_{.col}")) %>%
  ungroup() %>%
  group_by(Method) %>%
  summarise(across(starts_with("RANK_"), mean), .groups = "drop") %>%
  rename(id = Method)

# Preserve the right-hand six columns as the original per-dataset averages of
# NMI, ARI, AMI, and Overall; expanding the ranking display must not alter them.
performance_data <- raw %>%
  mutate(metric_average = rowMeans(across(all_of(METRIC_AVERAGE_METRICS)))) %>%
  select(Method, dataset, metric_average) %>%
  mutate(dataset = factor(dataset, levels = DATASET_ORDER)) %>%
  arrange(dataset) %>%
  mutate(dataset = unname(DATASET_LABEL[as.character(dataset)])) %>%
  pivot_wider(id_cols = Method, names_from = dataset,
              values_from = metric_average, names_prefix = "AVG_") %>%
  rename(id = Method)

method_info <- tibble(id = METHOD_ORDER)
dt <- method_info %>%
  left_join(rank_data, by = "id") %>%
  left_join(performance_data, by = "id") %>%
  mutate(
    SCORE_ARI = 1 - (RANK_ARI - 1) / 7,
    SCORE_NMI = 1 - (RANK_NMI - 1) / 7,
    SCORE_AMI = 1 - (RANK_AMI - 1) / 7,
    SCORE_FMI = 1 - (RANK_FMI - 1) / 7,
    SCORE_Homogeneity = 1 - (RANK_Homogeneity - 1) / 7,
    SCORE_Completeness = 1 - (RANK_Completeness - 1) / 7,
    `SCORE_V-measure` = 1 - (`RANK_V-measure` - 1) / 7,
    SCORE_Overall = 1 - (RANK_Overall - 1) / 7
  )

column_info <- tribble(
  ~id, ~group, ~name, ~geom, ~palette, ~options,
  "id", "Method", "Method", "text", "method", list(size = 4.2, width = 4.1),
  "SCORE_ARI", "Ranking score", "ARI", "bar", "rank", list(width = 3),
  "SCORE_NMI", "Ranking score", "NMI", "bar", "rank", list(width = 3),
  "SCORE_AMI", "Ranking score", "AMI", "bar", "rank", list(width = 3),
  "SCORE_FMI", "Ranking score", "FMI", "bar", "rank", list(width = 3),
  "SCORE_Homogeneity", "Ranking score", "Homogeneity", "bar", "rank", list(width = 3),
  "SCORE_Completeness", "Ranking score", "Completeness", "bar", "rank", list(width = 3),
  "SCORE_V-measure", "Ranking score", "V-measure", "bar", "rank", list(width = 3),
  "SCORE_Overall", "Ranking score", "Overall", "bar", "rank", list(width = 3),
  "AVG_P22", "Metric averages", "P22", "circle", "metric", list(width = 1.55),
  "AVG_E11", "Metric averages", "E11", "circle", "metric", list(width = 1.55),
  "AVG_E13", "Metric averages", "E13", "circle", "metric", list(width = 1.55),
  "AVG_E15", "Metric averages", "E15", "circle", "metric", list(width = 1.55),
  "AVG_E18", "Metric averages", "E18", "circle", "metric", list(width = 1.55),
  "AVG_Emb", "Metric averages", "Emb", "circle", "metric", list(width = 1.55)
)

column_groups <- tribble(
  ~level1, ~group, ~palette,
  "Method", "Method", "method",
  "Ranking score", "Ranking score", "rank",
  "Metric averages", "Metric averages", "metric"
)

palettes <- list(
  method = c("#f0f0f0", "#d9d9d9"),
  rank = brewer.pal(9, "Purples"),
  metric = brewer.pal(9, "YlOrBr")
)

legends <- list(
  list(palette = "method", enabled = FALSE),
  list(palette = "rank", geom = "bar", title = "Ranking score\nhigher = better",
       labels = c("0", rep("", 7), "1")),
  list(palette = "metric", geom = "circle", title = "Overall",
       labels = c("0", rep("", 7), "1"))
)

draw_panel <- function() {
  funky_heatmap(
    dt,
    column_info = column_info,
    column_groups = column_groups,
    palettes = palettes,
    legends = legends,
    add_abc = FALSE,
    col_annot_angle = 0
  )
}

pdf(OUT_PDF, width = 11.47, height = 3.45, useDingbats = FALSE)
draw_panel()
dev.off()

png(OUT_PNG, width = 3440, height = 1036, res = 300, bg = "white")
draw_panel()
dev.off()

cat("INPUT=", INPUT, "\nPDF=", OUT_PDF, "\nPNG=", OUT_PNG, "\n", sep = "")

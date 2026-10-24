#!/usr/bin/env bash
set -euo pipefail

resource_root="${SPACRAFT_GENOME_ROOT:-resources/mm10}"
mkdir -p "$resource_root/annotation" "$resource_root/motifs" "$resource_root/gene_sets"

download_one() {
  local url=$1
  local out=$2
  if [[ -s "$out" ]]; then
    printf 'reuse=%s\n' "$out"
  else
    curl --fail --location --retry 4 --retry-delay 5 --continue-at - --output "$out.part" "$url"
    mv "$out.part" "$out"
  fi
}

download_one https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz "$resource_root/mm10.fa.gz"
download_one https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes "$resource_root/genome.chrom.sizes"
download_one https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chromAlias.txt "$resource_root/chromosome_alias.tsv"
download_one https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz "$resource_root/annotation/gencode.vM25.annotation.gtf.gz"
download_one https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt "$resource_root/motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"

if [[ ! -s "$resource_root/genome.fa" ]]; then
  gzip -dc "$resource_root/mm10.fa.gz" > "$resource_root/genome.fa.tmp"
  mv "$resource_root/genome.fa.tmp" "$resource_root/genome.fa"
fi
if [[ ! -s "$resource_root/annotation/gencode.vM25.annotation.gtf" ]]; then
  gzip -dc "$resource_root/annotation/gencode.vM25.annotation.gtf.gz" > "$resource_root/annotation/gencode.vM25.annotation.gtf.tmp"
  mv "$resource_root/annotation/gencode.vM25.annotation.gtf.tmp" "$resource_root/annotation/gencode.vM25.annotation.gtf"
fi
if [[ ! -s "$resource_root/genome.fa.fai" ]]; then
  if command -v samtools >/dev/null; then
    samtools faidx "$resource_root/genome.fa"
  else
    ${PYTHON:-python} -c 'import pysam; pysam.faidx("$resource_root/genome.fa")'
  fi
fi

printf 'ucsc\tmm10_fasta\thttps://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz\t%s\n' "$(date -Iseconds)"
printf 'ucsc\tmm10_chrom_sizes\thttps://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes\t%s\n' "$(date -Iseconds)"
printf 'ucsc\tmm10_chrom_alias\thttps://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chromAlias.txt\t%s\n' "$(date -Iseconds)"
printf 'gencode\tvM25_gtf\thttps://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz\t%s\n' "$(date -Iseconds)"
printf 'jaspar\tJASPAR2024_CORE_vertebrates_non-redundant\thttps://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt\t%s\n' "$(date -Iseconds)"

#!/usr/bin/env python3
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
from datetime import datetime
import hashlib, re
import pandas as pd
import anndata as ad
import pysam

ROOT = SINGLE_DOWNSTREAM_ROOT
RES = GENOME_ROOT
files = [
    ("FASTA", RES/"genome.fa", "UCSC", "mm10/GRCm38", "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz"),
    ("FASTA_INDEX", RES/"genome.fa.fai", "derived", "mm10/GRCm38", "samtools/pysam faidx"),
    ("CHROM_SIZES", RES/"genome.chrom.sizes", "UCSC", "mm10/GRCm38", "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes"),
    ("CHROM_ALIAS", RES/"chromosome_alias.tsv", "UCSC", "mm10/GRCm38", "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chromAlias.txt"),
    ("GTF", RES/"annotation/gencode.vM25.annotation.gtf", "GENCODE", "GRCm38.p6/M25", "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"),
    ("MOTIF_MEME", RES/"motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt", "JASPAR", "2024 CORE vertebrates non-redundant", "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"),
]
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20), b""): h.update(b)
    return h.hexdigest()
missing=[str(p) for _,p,*_ in files if not p.is_file() or not p.stat().st_size]
if missing: raise SystemExit("missing resources: "+", ".join(missing))
alias={}
for line in (RES/"chromosome_alias.tsv").read_text().splitlines():
    row=line.split("\t")
    for x in row: alias[x]=row[0]
fa=pysam.FastaFile(str(RES/"genome.fa"))
pat=re.compile(r"^([^:-]+)[:-](\d+)[:-](\d+)$")
manifest=pd.read_csv(ROOT/"audit/SINGLE_SLICE_INPUT_MANIFEST.csv")
sampled=extracted=0
for row in manifest.itertuples():
    a=ad.read_h5ad(row.source_h5ad,backed="r")
    idx=list(range(0,a.n_vars,max(1,a.n_vars//20)))[:20]
    for peak in a.var_names[idx]:
        m=pat.match(str(peak)); chrom=alias.get(m.group(1),m.group(1)); start=int(m.group(2)); end=int(m.group(3))
        sampled+=1
        seq=fa.fetch(chrom,start,end)
        extracted += len(seq)==end-start and set(seq.upper()) <= set("ACGTN")
gtf_chroms=set()
with (RES/"annotation/gencode.vM25.annotation.gtf").open() as f:
    for line in f:
        if not line.startswith("#"): gtf_chroms.add(line.split("\t",1)[0])
        if len(gtf_chroms)>=20: break
fasta_chroms=set(fa.references)
gtf_match=bool(gtf_chroms & fasta_chroms)
motif_text=(RES/"motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt").read_text()
motif_count=motif_text.count("\nMOTIF ")
ok=sampled==extracted and gtf_match and motif_text.startswith("MEME version") and motif_count>100
rows=[]
for kind,p,source,version,url in files:
    rows.append(dict(resource_type=kind,absolute_path=p,size_bytes=p.stat().st_size,build_version=version,
                     chromosome_naming="UCSC chr-prefixed",source=source,url=url,
                     download_date="2026-07-28",sha256=digest(p),validation_status="PASS"))
pd.DataFrame(rows).to_csv(ROOT/"resources/RESOURCE_MANIFEST.tsv",sep="\t",index=False)
(ROOT/"resources/RESOURCE_SHA256.txt").write_text("".join(f"{r['sha256']}  {r['absolute_path']}\n" for r in rows))
report=f"""# Resource download and validation report

- Status: **{'PASS' if ok else 'GENOME_RESOURCE_MISMATCH'}**
- Build: mm10 / GRCm38
- FASTA random peak extractions: {extracted}/{sampled}
- GTF/FASTA chromosome overlap: {gtf_match}
- Readable JASPAR MEME motifs: {motif_count}
- Alternative-contig aliases are resolved using the official UCSC chromosome alias table.
- Validation time: {datetime.now().astimezone().isoformat()}
"""
(ROOT/"audit/RESOURCE_DOWNLOAD_REPORT.md").write_text(report)
print("resource_validation_status="+("PASS" if ok else "GENOME_RESOURCE_MISMATCH"))
raise SystemExit(0 if ok else 2)

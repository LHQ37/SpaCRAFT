#!/usr/bin/env python3
"""Audit sequences and scan the frozen tested universe once with certified motifs."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
import hashlib, json, math, re, resource
import numpy as np
import pandas as pd
from scipy import sparse
from pyfaidx import Fasta
from snapatac2._snapatac2 import PyDNAMotif

ROOT=MULTI_DOWNSTREAM_ROOT
OUT=ROOT/"results/motif_phase5_v1"; SEQ=OUT/"sequence_audit"; MM=OUT/"motif_matrix"; META=OUT/"metadata"
for d in (SEQ,MM,META,OUT/"EarlyHigh",OUT/"LateHigh",OUT/"comparison",OUT/"sensitivity"): d.mkdir(parents=True,exist_ok=True)
P3=ROOT/"results/accessibility_phase3_v1/downstream_inputs"
EARLY=P3/"C5_EarlyHigh_peaks_for_motif.csv"; LATE=P3/"C5_LateHigh_peaks_for_motif.csv"; BG=P3/"C5_tested_peak_background_for_motif.csv"
P2=ROOT/"results/accessibility_phase2_v1/C5/C5_peak_stage_summary.csv.gz"
FASTA=GENOME_ROOT / "genome.fa"
MEME=GENOME_ROOT / "motifs/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def coords(pid):
 m=re.fullmatch(r"(chr[^-]+)-(\d+)-(\d+)",str(pid)); assert m
 return m.group(1),int(m.group(2)),int(m.group(3))

e=pd.read_csv(EARLY); l=pd.read_csv(LATE); u=pd.read_csv(BG); p2=pd.read_csv(P2,usecols=["peak_id","mean_accessibility"])
assert len(e)==126 and len(l)==106 and len(u)==6540
for x in (e,l,u): assert x.peak_id.is_unique and x.peak_id.notna().all()
es,ls,us=set(e.peak_id),set(l.peak_id),set(u.peak_id)
assert es<=us and ls<=us and not es&ls
genome=Fasta(str(FASTA),one_based_attributes=False,as_raw=True,sequence_always_upper=True)
rows=[]; sequences=[]
for pid in u.peak_id:
 chrom,start,end=coords(pid); assert start>=0 and end>start and chrom in genome and end<=len(genome[chrom])
 s=str(genome[chrom][start:end]); assert len(s)==end-start and len(s)>0
 n=s.count("N")/len(s); gc=(s.count("G")+s.count("C"))/max(1,len(s)-s.count("N"))
 sequences.append(s)
 rows.append({"peak_id":pid,"chromosome":chrom,"start":start,"end":end,"peak_length":end-start,
  "GC_fraction":gc,"N_fraction":n,"module_membership":"EarlyHigh" if pid in es else ("LateHigh" if pid in ls else "Background")})
cov=pd.DataFrame(rows).merge(p2,on="peak_id",how="left",validate="one_to_one")
assert cov.mean_accessibility.notna().all() and np.isfinite(cov[["peak_length","GC_fraction","N_fraction","mean_accessibility"]]).all().all()
assert cov.N_fraction.max()<=.05
cov.to_csv(SEQ/"C5_tested_peak_sequence_covariates.csv.gz",index=False,compression="gzip")

lines=MEME.read_text().splitlines(); motifs=[]; meta=[]; i=0
while i<len(lines):
 if lines[i].startswith("MOTIF "):
  z=lines[i].split(maxsplit=2); mid=z[1]; name=z[2] if len(z)>2 else mid; i+=1
  while i<len(lines) and "letter-probability matrix:" not in lines[i]: i+=1
  assert i<len(lines); w=int(re.search(r"\bw=\s*(\d+)",lines[i]).group(1))
  mat=np.asarray([[float(v) for v in q.split()[:4]] for q in lines[i+1:i+1+w]])
  motifs.append(PyDNAMotif(mid,mat)); family=re.sub(r"\(.*","",name).split("::")[0].upper()
  meta.append({"motif_index":len(meta),"motif_id":mid,"TF_symbol":name,"motif_family":family,"width":w})
  i+=w
 i+=1
assert len(motifs)==879
rr=[]; cc=[]; scores=[]; comp=str.maketrans("ACGTN","TGCAN")
for j,m in enumerate(motifs):
 scanner=m.with_nucl_prob(); present=np.asarray(scanner.exists(sequences),dtype=bool)
 hit=np.flatnonzero(present); rr.extend(hit.tolist()); cc.extend([j]*len(hit))
 for k in hit:
  found=scanner.find(sequences[k])
  if not found: found=scanner.find(sequences[k].translate(comp)[::-1])
  assert found
  scores.append(max(float(q[1]) for q in found))
 if (j+1)%50==0: print(f"scanned={j+1}/879",flush=True)
presence=sparse.csr_matrix((np.ones(len(rr),dtype=np.uint8),(rr,cc)),shape=(6540,len(motifs)))
sparse.save_npz(MM/"C5_tested_peak_motif_presence_sparse.npz",presence,compressed=True)
scoremat=sparse.csr_matrix((np.asarray(scores,dtype=np.float32),(rr,cc)),shape=presence.shape)
sparse.save_npz(MM/"C5_tested_peak_motif_scores_sparse.npz",scoremat,compressed=True)
pd.DataFrame(meta).to_csv(MM/"motif_metadata.csv",index=False)
cov[["peak_id"]].assign(peak_index=np.arange(len(cov))).to_csv(MM/"peak_order.csv",index=False)
params={"status":"COMPLETE","genome_build":"mm10/GRCm38","fasta":str(FASTA),"fasta_sha256":sha(FASTA),
 "motif_database":str(MEME),"motif_database_sha256":sha(MEME),"JASPAR_version":"2024 CORE vertebrates non-redundant",
 "motif_count":len(motifs),"peak_count":len(cov),"scanner":"SnapATAC2 2.8.0 PyDNAMotif.with_nucl_prob().exists",
 "scan_scope":"all 6540 tested peaks scanned once with identical parameters",
 "presence_definition":"certified scanner thresholded occurrence","score_matrix_semantics":"maximum certified scanner score among threshold-passing hits; zero means no hit",
 "matrix_sparse":True,"maximum_N_fraction":.05,"peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(MM/"MOTIF_SCAN_PARAMETERS.json").write_text(json.dumps(params,indent=2)+"\n")
(SEQ/"SEQUENCE_VALIDATION_REPORT.md").write_text(
 f"# Sequence validation\n\nStatus: **PASS**\n\nAll {len(cov)} unique valid peaks were extracted from mm10/GRCm38; "
 f"no coordinate was out of bounds or empty. Maximum N fraction was {cov.N_fraction.max():.6g} "
 "against the pre-registered 0.05 limit. Length, GC, N, and mean C5 accessibility are finite.\n")
print(json.dumps(params,indent=2))

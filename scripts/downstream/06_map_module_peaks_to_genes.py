#!/usr/bin/env python3
"""Exact certified nearest-gene mapping for frozen Phase-3 peak sets."""
from pathlib import Path
from spacraft.paths import (
    ABLATION_ROOT, DATA_ROOT, FIGURE_OUTPUT_ROOT, GENOME_ROOT,
    MULTISLICE_RESULTS_ROOT, MULTI_DOWNSTREAM_ROOT, REPO_ROOT,
    SINGLE_DOWNSTREAM_ROOT,
)
from collections import defaultdict
import json, re
import numpy as np
import pandas as pd

ROOT=MULTI_DOWNSTREAM_ROOT; OUT=ROOT/"results/GO_phase6_v1"
MAP=OUT/"gene_mapping"; BG=OUT/"background"; META=OUT/"metadata"
for d in (MAP,BG,META,OUT/"EarlyHigh",OUT/"LateHigh",OUT/"comparison",OUT/"sensitivity"): d.mkdir(parents=True,exist_ok=True)
P3=ROOT/"results/accessibility_phase3_v1/downstream_inputs"; GTF=GENOME_ROOT / "annotation/gencode.vM25.annotation.gtf"
FILES={"EarlyHigh":P3/"C5_EarlyHigh_peaks_for_gene_mapping.csv","LateHigh":P3/"C5_LateHigh_peaks_for_gene_mapping.csv",
       "Background":P3/"C5_tested_peak_background_for_motif.csv"}
PROM_UP=2000; PROM_DOWN=500; MAX_DIST=100000
def attrs(s): return {m.group(1):m.group(2) for m in re.finditer(r'(\S+) "([^"]*)";',s)}
genes={}
allowed={f"chr{i}" for i in range(1,20)}|{"chrX","chrY"}
with GTF.open() as f:
 for line in f:
  if line.startswith("#"): continue
  z=line.rstrip().split("\t")
  if len(z)<9 or z[0] not in allowed or z[2] not in {"gene","exon","five_prime_UTR"}: continue
  a=attrs(z[8]); gid=a.get("gene_id","").split(".")[0]
  if not gid: continue
  if gid not in genes: genes[gid]={"gene_id":gid,"gene_symbol":a.get("gene_name",gid),"chromosome":z[0],"start":int(z[3])-1,"end":int(z[4]),"strand":z[6],"exons":[],"utr5":[]}
  g=genes[gid]
  if z[2]=="gene": g.update(gene_symbol=a.get("gene_name",gid),chromosome=z[0],start=int(z[3])-1,end=int(z[4]),strand=z[6])
  elif z[2]=="exon": g["exons"].append((int(z[3])-1,int(z[4])))
  else: g["utr5"].append((int(z[3])-1,int(z[4])))
genes=[g for g in genes.values() if g["end"]>g["start"]]
bychr=defaultdict(list)
for i,g in enumerate(genes):
 g["tss"]=g["start"] if g["strand"]=="+" else g["end"]-1; bychr[g["chromosome"]].append(i)
idx={}
for chrom,ii in bychr.items():
 ii=np.asarray(ii); o=np.argsort([genes[i]["tss"] for i in ii]); ii=ii[o]; idx[chrom]=(np.asarray([genes[i]["tss"] for i in ii]),ii)
def overlap(s,e,ints): return any(s<b and e>a for a,b in ints)
def one(pid,module):
 m=re.fullmatch(r"(chr[^-]+)-(\d+)-(\d+)",str(pid))
 if not m: return {"peak_id":pid,"chromosome":"","start":np.nan,"end":np.nan,"annotation":"unannotated","gene_id":"","gene_symbol":"","distance_to_TSS":np.nan,"mapping_rule":"UNMAPPED_PARSE","module":module}
 chrom,s,e=m.group(1),int(m.group(2)),int(m.group(3)); mid=(s+e)//2
 if chrom not in idx: return {"peak_id":pid,"chromosome":chrom,"start":s,"end":e,"annotation":"unannotated","gene_id":"","gene_symbol":"","distance_to_TSS":np.nan,"mapping_rule":"UNMAPPED_NONSTANDARD_CHROMOSOME","module":module}
 tss,ii=idx[chrom]; k=np.searchsorted(tss,mid); cand=np.unique(np.clip(np.arange(k-4,k+5),0,len(ii)-1))
 gi=int(ii[cand[np.argmin(np.abs(tss[cand]-mid))]]); g=genes[gi]; signed=(mid-g["tss"])*(1 if g["strand"]=="+" else -1)
 if -PROM_UP<=signed<=PROM_DOWN: ann="promoter"
 elif overlap(s,e,g["utr5"]): ann="5' UTR"
 elif overlap(s,e,g["exons"]): ann="exon"
 elif s<g["end"] and e>g["start"]: ann="intron"
 elif 0<signed<=2000: ann="downstream"
 else: ann="distal/intergenic"
 direct=ann in {"promoter","5' UTR","exon","intron"}; keep=direct or abs(signed)<=MAX_DIST
 rule=("DIRECT_PROMOTER_OR_GENE_BODY" if direct else ("NEAREST_GENE_WITHIN_100KB" if keep else "UNMAPPED_BEYOND_100KB"))
 return {"peak_id":pid,"chromosome":chrom,"start":s,"end":e,"annotation":ann,
  "gene_id":g["gene_id"] if keep else "","gene_symbol":g["gene_symbol"] if keep else "",
  "distance_to_TSS":signed,"mapping_rule":rule,"module":module}

tabs={}
for module,path in FILES.items():
 x=pd.read_csv(path); expected={"EarlyHigh":126,"LateHigh":106,"Background":6540}[module]
 assert len(x)==expected and x.peak_id.is_unique
 tab=pd.DataFrame([one(p,module) for p in x.peak_id]); tabs[module]=tab
 name={"EarlyHigh":"EarlyHigh_peak_to_gene.csv","LateHigh":"LateHigh_peak_to_gene.csv","Background":"tested_background_peak_to_gene.csv"}[module]
 tab.to_csv(MAP/name,index=False)
 if module!="Background":
  mapped=tab[tab.gene_id.ne("")]
  gt=(mapped.groupby(["gene_id","gene_symbol"]).agg(peak_count=("peak_id","size"),
      promoter_gene_body_peak_count=("annotation",lambda s:s.isin(["promoter","5' UTR","exon","intron"]).sum()),
      distal_peak_count=("mapping_rule",lambda s:s.eq("NEAREST_GENE_WITHIN_100KB").sum()),
      peak_ids=("peak_id",lambda s:";".join(sorted(s)))).reset_index())
  gt.to_csv(OUT/f"{module}_module_associated_gene_table.csv",index=False)
  (OUT/f"{module}_module_associated_genes.txt").write_text("\n".join(sorted(gt.gene_symbol))+"\n")
bg=tabs["Background"]; mapped=bg[bg.gene_id.ne("")]
bgt=(mapped.groupby(["gene_id","gene_symbol"]).agg(peak_count=("peak_id","size"),
 mapping_rules=("mapping_rule",lambda s:";".join(sorted(set(s)))),peak_ids=("peak_id",lambda s:";".join(sorted(s)))).reset_index())
bgt.to_csv(BG/"C5_tested_peak_background_gene_table.csv",index=False)
(BG/"C5_tested_peak_background_genes.txt").write_text("\n".join(sorted(bgt.gene_symbol))+"\n")
summary={}
for m,t in tabs.items():
 summary[m]={"peak_count":len(t),"mapped_peak_count":int(t.gene_id.ne("").sum()),"unmapped_peak_count":int(t.gene_id.eq("").sum()),
  "unique_gene_count":int(t.loc[t.gene_id.ne(""),"gene_id"].nunique()),"annotation_counts":t.annotation.value_counts().to_dict(),
  "mapping_rule_counts":t.mapping_rule.value_counts().to_dict()}
(META/"peak_to_gene_mapping_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

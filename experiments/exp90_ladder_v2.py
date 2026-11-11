#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9-1 LADDER V2b (после audit2 B1; v2 умер в CI-баге np.median(2D без axis=1)).
Три статистики на (blk):
  masked   — Δ(1) по внутриблочным парам шаффленной последовательности;
  unmasked — по всем парам шаффленной;
  control  — по СЛУЧАЙНОМУ подмножеству пар того же размера из НЕшаффленной (позиционный
             контроль: маска отбирает не случайные позиции, а, напр., чётные при blk=2).
Trivial-предиктор B1: unmasked ≈ (1−1/b)·masked. seam_delta = unmasked − trivial:
≈0 ⇒ лестница была разбавлением (B1 подтверждён); <0 (уходит в ноль сильнее) ⇒ шовные
пары несут антидип (активное разрушение); >0 ⇒ швы несут избыток dip'а (дальнодействие).
САМОАУДИТ v2 (записан в ТЗ): masked НЕ равно real тождественно — it есть детерминированный
позиционный подотбор; для blk=2 это чётные лаги → control обязателен, чтобы отделять
«носителя нет в нечётных позициях» от «шаффл убил».
Референс real — ЭТОТ ЖЕ оценщик на исходном порядке (t25_ekurve.json потерян).
loopy — max Δ(k≥2)>0.95 по real-кривии этого скрипта + исторические wiki_0/wiki_4.
"""
import json, os, zlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp62_batt_lib import curve_u
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"
SEEDS=[1,2,3,4,5]; BLKS=[2,4,8,16,32,64]; NC=12   # NC — число контрольных подотборов
z=np.load(NPZ,allow_pickle=True)
meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
def cos1(P):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.einsum("ij,ij->i",Pn[:-1],Pn[1:])
def dcurve_full(P):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12); t=np.arange(len(P)); out=[]
    for k in range(1,97):
        c=np.einsum("ij,ij->i",Pn[:-k],Pn[k:]); out.append(float(np.median(c)))
    return np.array(out)
def masked_unmasked(c1,blk):
    t=np.arange(len(c1)); m=(t//blk)==((t+1)//blk)
    return float(np.median(c1[m])), float(np.median(c1)), float(m.mean())
def blk_path(u,hn,seed,blk):
    r=np.random.default_rng(seed); T=len(u); nb=T//blk; idx=r.permutation(nb)
    U=np.concatenate([u[b*blk:(b+1)*blk] for b in idx],0); H=np.concatenate([hn[b*blk:(b+1)*blk] for b in idx],0)
    if nb*blk<T: U=np.concatenate([U,u[nb*blk:]]); H=np.concatenate([H,hn[nb*blk:]])
    return U,H
real={}; c1real={}; loopy=set()
for nm in names:
    u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64); hn=np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)
    P=curve_u(u,hn); real[nm]=dcurve_full(P); c1real[nm]=cos1(P)
    if float(np.max(real[nm][1:]))>0.95: loopy.add(nm)
loopy|={"wiki_0","wiki_4"}
ok=[n for n in names if n not in loopy]; nw=[n for n in ok if dom[n]!="wiki"]
print(f"docs={len(names)} nonloopy={len(ok)} nonwiki={len(nw)}",flush=True)
rngc=np.random.default_rng(4242)
per={b:{} for b in BLKS}          # (nm,sd) -> dict(masked,unmasked,control,frac)
for nm in ok:
    u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64); hn=np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)
    for b in BLKS:
        for sd in SEEDS:
            U,H=blk_path(u,hn,zlib.crc32(f"{nm}|{b}|{sd}".encode())%2**32,b)
            c1=cos1(curve_u(U,H))
            msk,unm,frac=masked_unmasked(c1,b)
            # контроль ПОЗИЦИОННОГО отбора: тот же размер подотбора, но из НЕшаффленного c1 (real)
            c1r=c1real[nm]; msk_sz=max(int(frac*len(c1)),2)
            ctrl=[float(np.median(c1r[rngc.choice(len(c1r),min(msk_sz,len(c1r)),replace=False)])) for _ in range(NC)]
            per[b][(nm,sd)]=dict(masked=msk,unmasked=unm,control=float(np.median(ctrl)))
def medsel(b,key,sel):
    v=[per[b][k][key] for k in per[b] if k[0] in sel]
    return float(np.median(v)) if v else float("nan")
rng=np.random.default_rng(123)
def cluster_boot(b,sel):
    # docs-кластеры: медиана delta по сидам внутри дока, bootstrap по докам (axis=1!)
    docs=sorted({k[0] for k in per[b] if k[0] in sel}); M=[]
    for n in docs:
        ds=[per[b][(n,sd)]["unmasked"]-(1-1/b)*per[b][(n,sd)]["masked"] for sd in SEEDS if (n,sd) in per[b]]
        M.append(np.median(ds))
    M=np.array(M)
    bs=np.median(M[rng.integers(0,len(M),(2000,len(M)))],axis=1)
    return float(np.median(M)),[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)],len(M)
out={"meta":{"n_docs":len(names),"nonloopy":ok,"seeds":SEEDS,"L":L,
     "note":"dip k=1; hump/plateau лестницей с маской НЕ измеримы (нет пар k>=blk); масштаб сверх локальности — R9-6 surrogate"},
     "real_pooled":round(float(np.median([float(np.median(c1real[n])) for n in nw])),4)}
tab={}
for b in BLKS:
    sel=nw
    row=dict(real=round(float(np.median([np.median(c1real[n]) for n in sel])),4),
             masked=round(medsel(b,"masked",sel),4),unmasked=round(medsel(b,"unmasked",sel),4),
             control=round(medsel(b,"control",sel),4))
    v,ci,n=cluster_boot(b,sel); row["seam_delta"]=round(v,4); row["ci"]=ci; row["n_docs"]=n
    tab[f"blk{b}"]=row
out["dip_table_pooled_nonwiki"]=tab
os.makedirs(ROOT+"/results/e90",exist_ok=True)
json.dump(out,open(ROOT+"/results/e90/LADDER_V2.json","w"),indent=1,ensure_ascii=False)
print(f"\n{'blk':>5} {'real':>8} {'masked':>8} {'control':>8} {'unmasked':>9} {'trivial':>8} {'seam_delta':>10} CI")
for b in BLKS:
    r=tab[f"blk{b}"]; tr=round((1-1/b)*r["masked"],4)
    print(f"{b:>5} {r['real']:>8} {r['masked']:>8} {r['control']:>8} {r['unmasked']:>9} {tr:>8} {r['seam_delta']:>10} {r['ci']}")
print("\nЧТЕНИЕ: masked≈control ⇒ позиционный отбор не важен; unmasked≈trivial ⇒ B1 (лестница=разбавление);"
      "\nsign(seam_delta)≠0 с CI∌0 — реальный вклад швов (дальность).",flush=True)
print("R9-1 DONE",flush=True)

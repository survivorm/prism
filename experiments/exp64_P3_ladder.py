#!/usr/bin/env python
"""S5 личное P3: полный блочный лестничный прогон кривых k=1..96 (blk 2/4/8/16/32/64) —
на каком масштабе рассыпается горб и плато. Плюс P3b: длина предложения в токенах по доменам
(гипотеза «32 = предложение пула», не модель).
САМОАУДИТ: перемешиваем u и hn БЛОКАМИ вместе (тот же путь, что t25 blk32); seed crc32(nm|blk);
невырожденные 36 (wiki=1 — только в пул non-wiki); фичи: dip=d(1), горб=max d(2..8), плато=median d(9..96);
bootstrap CI 2000; sentence: id чьи строки коротки и кончаются на .!? — грубый прокси, зафлаган."""
import json, os, sys, zlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np, sys as _s; _s.path.insert(0,'.')
from exp62_batt_lib import curve_u
from transformers import AutoTokenizer
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"; KM=96
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
def blk_path(u,hn,seed,blk):
    r=np.random.default_rng(zlib.crc32(seed.encode())%99991); T=len(u); nb=T//blk; idx=r.permutation(nb)
    U=np.concatenate([u[b*blk:(b+1)*blk] for b in idx],0); H=np.concatenate([hn[b*blk:(b+1)*blk] for b in idx],0)
    if nb*blk<T: U=np.concatenate([U,u[nb*blk:]]); H=np.concatenate([H,hn[nb*blk:]])
    return U,H
def dcurve(P,kk):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
rng=np.random.default_rng(77)
BLKS=[2,4,8,16,32,64]
cur={b:{} for b in BLKS}
for i,nm in enumerate(names):
    u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64); hn=np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)
    for b in BLKS:
        U,H=blk_path(u,hn,f"{nm}|{b}",b)
        cur[b][nm]=dcurve(curve_u(U,H),np.arange(1,KM+1))
    if i%16==15: print("P3",i+1,flush=True)
def feats(c): return np.array([c[0], c[1:8].max(), np.median(c[8:96])])
out={"features":["dip","hump","plateau"],"by_blk":{}}
nw=[n for n in ok if dom[n]!="wiki"]
for b in BLKS:
    F={dm:[feats(cur[b][n]) for n in ok if dom[n]==dm] for dm in ["math","code","chat"]}
    F["pooled_nonwiki"]=[feats(cur[b][n]) for n in nw]
    row={}
    for k,Fv in F.items():
        A=np.array(Fv); med=np.median(A,0)
        bs=np.median(A[rng.integers(0,len(A),(2000,len(A)))],0)
        row[k]={"med":[round(float(x),4) for x in med],
                "ci":[[round(float(bs[:,j].min()),4),round(float(bs[:,j].max()),4)] for j in range(3)]}
    out["by_blk"][b]=row
# reference real (t25)
Fr=np.array([feats(t25["curves_k1_96"]["real"][n]) for n in nw])
out["real_pooled_nonwiki"]=[round(float(x),4) for x in np.median(Fr,0)]
# P3b sentence scale
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
allids=sorted({int(t) for n in names for t in np.asarray(z[f"{n}_tok"]).ravel().tolist()})
SENT=set()
for i in allids:
    s=tk.convert_ids_to_tokens(i)
    if isinstance(s,str) and 1<=len(s)<=5 and s.rstrip('Ġ▁').endswith(('.','!','?')) and not s.rstrip('Ġ▁').strip('.!?').isdigit(): SENT.add(i)
gaps={}
for dm in ["wiki","math","code","chat"]:
    G=[]
    for n in [x for x in names if dom[x]==dm]:
        toks=np.asarray(z[f"{n}_tok"]).ravel()
        pos=np.flatnonzero(np.isin(toks,list(SENT)))
        if len(pos)>2: G.extend(np.diff(pos).tolist())
    G=np.array(G) if G else np.zeros(0)
    gaps[dm]={"n":len(G),"med":int(np.median(G)) if len(G) else None,"q25":int(np.percentile(G,25)) if len(G) else None,"q75":int(np.percentile(G,75)) if len(G) else None,"frac_le32":round(float((G<=32).mean()),2) if len(G) else None}
out["sentence_gaps"]=gaps
print(json.dumps({k:out[k] for k in ["real_pooled_nonwiki","sentence_gaps"]},ensure_ascii=False))
for b in BLKS:
    print("blk",b,{k:v["med"] for k,v in out["by_blk"][b].items()})
json.dump(out,open("../results/e64/P3_ladder.json","w"),ensure_ascii=False)
print("saved")

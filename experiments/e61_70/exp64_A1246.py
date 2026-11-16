#!/usr/bin/env python
"""E-64 A-блок ч.1: A1 space-matched null дипа; A2 норм-канал; A4 null градиента нити_Г+C1;
A6 состав/доменные строки на невырожденных. САМОАУДИТ-ЗАМЕТКИ:
- nullы: WHITE-STATE (iid строки h из маргинала дока ⇒ Δ MA(1) ⇒ базлайн ≈−0.5) и
  RW-INCREMENT (iid строки Δh ⇒ cos пары ≈0). «Базовый слой» любого Δ-оценщика — между ними.
- Pearson кривых норм НЕ центрируется дважды; для hn(level) центрирование по доку.
- все median-агрегации: per-doc медиана → domain median (Б8)."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
sys.path.insert(0,'.')
from exp62_batt_lib import curve_u
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
rng=np.random.default_rng(2024)
out={}

# ============ A1: dip по геометриям, два null-базлайна ============
def med_cos_pairs(A,B):
    a=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-20)
    b=B/np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-20)
    return float(np.median(np.einsum("ij,ij->i",a,b)))
g_raw={};g_emb={};g_V={}
for i,nm in enumerate(names):
    h=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64)*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)[:,None]
    Dh=np.diff(h,axis=0)
    g_raw[nm]={"real":med_cos_pairs(Dh[:-1],Dh[1:]),
      "nullW":med_cos_pairs(h[rng.integers(0,len(h),20000)]-h[rng.integers(0,len(h),20000)],
                            h[rng.integers(0,len(h),20000)]-h[rng.integers(0,len(h),20000)]),
      "nullRW":med_cos_pairs(Dh[rng.integers(0,len(Dh),20000)],Dh[rng.integers(0,len(Dh),20000)])}
    g_emb[nm]={"real":t25  and None} # placeholder replaced below
    if i%16==15: print("A1",i+1,flush=True)
e64ab=json.load(open("../results/e64/e64ab_quantizer.json"))
emb_curv=e64ab["emb"]
for nm in names:
    g_emb[nm]={"real":emb_curv["per_doc"][nm][0],
               "nullW":float(np.median([emb_curv["shuf_by_dom"][dom[nm]][0]]*1))}
for nm in names:
    V=curve_u(z[f"{nm}_L{L}_u"],z[f"{nm}_L{L}_hn"])
    g_V[nm]={"real":float(np.median(np.einsum("ij,ij->i",V[:-1],V[1:]))),
             "nullW":0.0,  # iid-sphere walk: T-17 = 0.000 (MA-free оценщик)
             "nullRW":None,
             "nullBLK32":t25["curves_k1_96"]["blk32"][nm][0]}
def agg(d,dm=None,keys=None):
    ns=[n for n in (ok if dm=="ok" else names) if dm in(names,ok) or dom[n]==dm] if dm not in("ALL","ok") else [n for n in names if dm=="ALL" or n in ok]
    r={}
    for k in keys:
        v=[d[n][k] for n in ns if d[n].get(k) is not None]
        r[k]=round(float(np.median(v)),3) if v else None
    return r
A1={}
for geo,d in [("raw_h",g_raw),("emb",g_emb),("V",g_V)]:
    ks=[k for k in list(d.values())[0].keys()]
    A1[geo]={scope:agg(d,scope,ks) for scope in ["ALL","ok","wiki","math","code","chat"]}
out["A1_dip_geometries"]=A1
print("A1 raw_h:",A1["raw_h"]["ok"]); print("A1 V:",A1["V"]["ok"])

# ============ A2: норм-канал ============
def pearson_series(x,k):
    x=x-x.mean(); d=np.sqrt((x@x)*(x[:len(x)-k]@x[:len(x)-k])+1e-30)
    return float((x[:len(x)-k]@x[k:])/d)
KM=96
def norm_curves(hn):
    d=np.diff(hn)
    return ([pearson_series(d,k) for k in range(1,KM+1)],
            [pearson_series(hn,k) for k in range(1,KM+1)])
def blk_rows(x,seed,blk=32):
    r=np.random.default_rng(seed); T=len(x); nb=T//blk; idx=r.permutation(nb)
    o=np.concatenate([x[b*blk:(b+1)*blk] for b in idx],0)
    return np.concatenate([o,x[nb*blk:]]) if nb*blk<T else o
nc={};ncb={}
for nm in names:
    hn=np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)
    c,_=norm_curves(hn); nc[nm]=c
    cb,_=norm_curves(blk_rows(hn,hash(nm)%99991)); ncb[nm]=cb
def dmed(dct,dm,k):
    ns=[n for n in ok if dom[n]==dm]
    return round(float(np.median([dct[n][k-1] for n in ns])),3)
out["A2_norm"]={"inc_by_dom":{dm:[dmed(nc,dm,k) for k in (1,2,4,8,16,32,48,64,96)] for dm in ["wiki","math","code","chat"]},
                "inc_k":[1,2,4,8,16,32,48,64,96],
                "blk_inc_by_dom":{dm:[dmed(ncb,dm,k) for k in (1,2,4,8,16,32,48,64,96)] for dm in ["wiki","math","code","chat"]},
                "level_by_dom":{dm:[round(float(np.median([norm_curves(np.asarray(z[f'{n}_L{L}_hn'],dtype=np.float64))[1][k-1] for n in ok if dom[n]==dm])),3) for k in (1,2,4,8,16,32,64,96)] for dm in ["wiki","math","code","chat"]}}
print("A2 inc:",out["A2_norm"]["inc_by_dom"])

# ============ A4: градиент нити_Г: null + C1 ============
LAY=["1","5","11","17","23"]; PAIRS=[("1","5"),("5","11"),("11","17"),("17","23")]
def H(n,L): return np.asarray(z[f"{n}_L{L}_u"],dtype=np.float64)*np.asarray(z[f"{n}_L{L}_hn"],dtype=np.float64)[:,None]
def nrm(g): return g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
gs={(a,b):{n:nrm(H(n,b)-H(n,a)) for n in names} for a,b in PAIRS}
A4={}
for a,b in PAIRS:
    real={n:np.median(np.einsum("ij,ij->i",gs[(a,b)][n][:-1],gs[(a,b)][n][1:])) for n in ok}
    # null-1: кросс-док (тот же домен, разные документы/времена)
    nullcd=[]
    for rep in range(20):
        for n in ok:
            pool=[m for m in ok if dom[m]==dom[n] and m!=n]
            m=pool[rng.integers(len(pool))]; off=rng.integers(len(gs[(a,b)][n]))
            g1=gs[(a,b)][n]; g2=gs[(a,b)][m]
            idx=rng.integers(0,len(g1),300); j=(idx+off)%len(g2)
            nullcd.append(np.median(np.einsum("ij,ij->i",g1[idx],g2[j])))
    # null-2: large-shift внутри дока
    nullsh=[]
    for rep in range(20):
        for n in ok:
            g=gs[(a,b)][n]; s=rng.integers(40,len(g)-100)
            idx=rng.integers(0,len(g)-s-50,300)
            nullsh.append(np.median(np.einsum("ij,ij->i",g[idx],g[idx+s:s+idx+1][np.arange(300)% (len(g)-s)] if False else g[idx+s:idx+s+1][0:1]*(0+1)+g[idx]*0+g[idx+s]*0+ (g[idx]*0+g[idx+s:idx+s+1])[(np.arange(300))% max(len(g)-s-idx.max() if False else 1,1)]] if False else np.einsum("ij,ij->i",g[idx],np.take(g,idx+s,mode="clip"))))
    A4[f"{a}->{b}"]={"real_by_dom":{dm:round(float(np.median([real[n] for n in ok if dom[n]==dm])),3) for dm in ["wiki","math","code","chat"]},
        "real_med":round(float(np.median(list(real.values()))),3),
        "null_crossdoc":round(float(np.median(nullcd)),3),
        "null_bigshift":round(float(np.median(nullsh)),3)}
L23=json.load(open("../results/e62/layers_qwen.json"))
dip={lay:L23[lay]["dip1"] for lay in ["L1","L5","L11","L17","L23"]}
gx=[];dy=[]
for a,b in PAIRS:
    mid=b; gx.append(A4[f"{a}->{b}"]["real_med"]); dy.append(dip[f"L{mid}"])
A4["C1_spear_n4"]={"g":gx,"dip":dy,
  "spearman":round(float(np.corrcoef(np.argsort(np.argsort(gx)),np.argsort(np.argsort(dy)))[0,1]),2)}
out["A4_thread_grad"]=A4
print("A4:",{k:A4[k] for k in PAIRS and [f'{a}->{b}' for a,b in PAIRS]}["1->5"],"...",A4["C1_spear_n4"])

# ============ A6: состав + доменные медианы на невырожденных ============
mark=json.load(open("../results/e64/e64c_markov.json"))["per_doc"]
A6={}
for dm in ["wiki","math","code","chat"]:
    ns=[n for n in ok if dom[n]==dm]
    oms=[mark[n]["om"] for n in ns if mark[n]["win"]=="osc"]
    A6[dm]={"n_nondeg":len(ns),"osc_frac":round(len(oms)/max(len(ns),1),2),
      "om_med_osc":round(float(np.median(oms)),2) if oms else None,
      "joint4_med":round(float(np.median([mark[n]["joint4"]*mark[n]["cand_frac"] for n in ns])),2),
      "E33_96_med":round(float(np.median([np.mean(np.array(t25["curves_k1_96"]["real"][n][32:96])-np.array(t25["curves_k1_96"]["blk32"][n][32:96])) for n in ns])),4)}
A6["ALL_nondeg"]={"n":len(ok)}
out["A6_composition"]=A6
print("A6:",json.dumps(A6,ensure_ascii=False))
json.dump(out,open("../results/e64/A1246.json","w"),ensure_ascii=False)
print("saved A1246.json")

#!/usr/bin/env python
"""E-63 T-25: избыточная кривая E(k) + дальнодействие ПОСЛЕ блока.
Аудит-вопрос: r2(null-лестницы)=0.98 при blk16/32 >= real 0.96; fit батареи видит
только k<=24. Вопрос «избыточное дальнодействие» = живёт ли Δ(k) при k>32 выше
blk32-нуля (тот убивает весь порядок >32 по построению). Плюс E(k) на удлинённой
сетке — объект из S1§3.1. Агрегация: медиана per-doc, затем по доменам (Б8)."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
sys.path.insert(0,'.')
from exp62_batt_lib import curve_u  # тот же оценщик касательных

NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"
z=np.load(NPZ, allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(meta[n]["dom"]),int(n.rsplit("_",1)[-1])))
dom={n:meta[n]["dom"] for n in names}
KM=96

def U_of(nm):
    u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64)
    U=u*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)[:,None]
    return U/np.maximum(np.linalg.norm(U,axis=1,keepdims=True),1e-12)

def curve_V(V,kmax=KM):
    out=[]
    for k in range(1,kmax+1):
        out.append(float(np.median(np.einsum("ij,ij->i",V[:-k],V[k:]))))
    return np.array(out)

def null_block(U,seed,blk):
    rng=np.random.default_rng(seed); T=len(U); nb=T//blk; idx=rng.permutation(nb)
    out=np.concatenate([U[b*blk:(b+1)*blk] for b in idx],0)
    if nb*blk<T: out=np.vstack([out,U[nb*blk:]])
    return out
def null_iid(U,seed):
    rng=np.random.default_rng(seed); T,D=U.shape
    d=np.median(np.linalg.norm(np.diff(U,axis=0),axis=1)); out=np.empty_like(U); out[0]=U[0]
    for t in range(1,T):
        n=rng.standard_normal(D); n-=n@out[t-1]*out[t-1]
        out[t]=out[t-1]+d*n/max(np.linalg.norm(n),1e-12); out[t]/=np.linalg.norm(out[t])
    return out
def null_arneg(U,seed,phi=-0.07):
    rng=np.random.default_rng(seed); T,D=U.shape
    out=np.empty_like(U); out[0]=U[0]
    for t in range(1,T):
        xi=rng.standard_normal(D); xi-=xi@out[t-1]*out[t-1]; xi/=np.linalg.norm(xi)
        v=phi*out[t-1]+np.sqrt(1-phi**2)*xi; out[t]=v/np.linalg.norm(v)
    return out

fams={"real":lambda U,i:U,"blk16":lambda U,i:null_block(U,i+5001,16),
      "blk32":lambda U,i:null_block(U,i+5002,32),"iid":lambda U,i:null_iid(U,i+5003),
      "arneg":lambda U,i:null_arneg(U,i+5004)}
curves={f:{} for f in fams}
for i,nm in enumerate(names):
    U=U_of(nm)
    for f,fn in fams.items():
        curves[f][nm]=curve_V(curve_u(fn(U,i),None))
    if i%16==15: print("...",i+1,flush=True)

def agg(f,dm=None):
    ns=[n for n in names if dm is None or dom[n]==dm]
    M=np.array([curves[f][n] for n in ns])          # docs x lags
    return np.median(M,axis=0)                      # median of per-doc medians

def loopy(nm):  # скрытый/явный луп: Δ(k)≈1 при k>=2 (аудит §4.8.3)
    return bool(np.max(curves["real"][nm][1:])>0.85)

res={}
ok=[n for n in names if not loopy(n)]
print("лупо-подобных docs:",[n for n in names if loopy(n)])
rng=np.random.default_rng(7)
out={}
for dm in ["all"]+["wiki","math","code","chat"]:
    ns=[n for n in ok if dm=="all" or dom[n]==dm]
    Rr=np.array([curves["real"][n] for n in ns]); Rb=np.array([curves["blk32"][n] for n in ns])
    E=Rr-Rb  # docs x lags
    boot=[E[rng.integers(0,len(ns),(len(ns),))].mean(axis=0).mean() for _ in range(2000)]
    seg=[(25,32),(33,48),(49,64),(65,96)]
    segs={}
    for a,b in seg:
        segv=E[:,a-1:b].mean(axis=0)  # per-doc mean over segment
        bs=[segv[rng.integers(0,len(ns),len(ns))].mean() for _ in range(2000)]
        segs[f"{a}-{b}"]=[float(np.median(segv)),float(np.quantile(bs,0.025)),float(np.quantile(bs,0.975))]
    out[dm]={"n":len(ns),"E_med_by_k":{str(k+1):float(np.median(E[:,k])) for k in [31,47,63,95]},
             "seg":segs}
    print(dm,"E(real-blk32) med by seg:",segs,flush=True)
res={"seg_E":out,"curves_k1_96":{f:{n:list(np.round(curves[f][n],4)) for n in names} for f in fams}}
json.dump(res,open("../results/e63/t25_ekurve.json","w"))
print("saved t25_ekurve.json")

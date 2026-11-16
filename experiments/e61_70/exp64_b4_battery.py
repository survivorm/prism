#!/usr/bin/env python
"""E-64 B4-батарея по Kaggle-пулам T=0.3/0.7/1.0 (pre-reg S1-6.6): copy обязана упасть,
знак дипа и ≤32-локальность — выжить. Для каждого T: degeneracy(loopy), кривые real/blk32
(dip/hump/plateau), lk4 copy rate, spearman(hit~p1 недоступен — нет маржин; только hit-структура).
САМОАУДИТ: T=0 база берётся из того же кода на gen_layers_qwen.npz (не из t25-констант);
blk32 тот же генератор (crc32 seed); wiki=1 фильтр не влияет на пул-фичи — пишем n_nondeg."""
import json, os, sys, zlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np, sys as _s; _s.path.insert(0,'.')
def dcurve(P,kk):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def Vpath(u,hn):
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    Vs=[]
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        if np.linalg.norm(vv)>1e-6: Vs.append(vv/np.linalg.norm(vv))
    return np.array(Vs)
def blk_path(u,hn,seed,blk=32):
    r=np.random.default_rng(zlib.crc32(seed.encode())%99991); T=len(u); nb=T//blk; idx=r.permutation(nb)
    U=np.concatenate([u[b*blk:(b+1)*blk] for b in idx],0); H=np.concatenate([hn[b*blk:(b+1)*blk] for b in idx],0)
    if nb*blk<T: U=np.concatenate([U,u[nb*blk:]]); H=np.concatenate([H,hn[nb*blk:]])
    return U,H
def lk4(toks):
    T=len(toks); nxt={}; hit=np.zeros(T,dtype=bool)
    for t in range(3,T-1):
        key=(int(toks[t-3]),int(toks[t-2]),int(toks[t-1]),int(toks[t]))
        pv=nxt.get(key)
        if pv is not None and pv==int(toks[t+1]): hit[t]=True
        nxt[key]=int(toks[t+1])
    return hit
def battery(npz_path,meta_path,tag):
    z=np.load(npz_path); meta=json.load(open(meta_path))
    dom={n:meta["docs"][n]["dom"] for n in meta["docs"]}
    names=sorted(meta["docs"],key=lambda n:(["wiki","math","code","chat"].index(dom[n]),meta["docs"][n]["si"]))
    rows=[];hitf=[];loopy=0
    KK=np.arange(1,97)
    for nm in names:
        u=z[f"{nm}_L23_u"].astype(np.float64); hn=z[f"{nm}_L23_hn"].astype(np.float64)
        cr=dcurve(Vpath(u,hn),KK)
        Ub,Hb=blk_path(u,hn,nm+"|T")
        cb=dcurve(Vpath(Ub,Hb),KK)
        toks=z[f"{nm}_tok"]
        h=lk4(toks)[4:383].mean()
        lp=bool(cr[1:].max()>0.95); loopy+=lp
        rows.append({"nm":nm,"dom":dom[nm],"loopy":lp,"dip":cr[0],"hump":cr[1:8].max(),
                     "plateau":float(np.median(cr[8:96])),"E_med":float(np.median(cr-cb)),
                     "hit":round(float(h),3)})
        hitf.append(float(h))
    ok=[r for r in rows if not r["loopy"]]
    def md(f,sub=None):
        v=[f(r) for r in (ok if sub is None else sub)]
        return round(float(np.median(v)),4) if v else None
    out={"tag":tag,"n":len(rows),"n_nondeg":len(ok),"loopy_frac":round(loopy/len(rows),3),
         "dip":md(lambda r:r["dip"]),"hump":md(lambda r:r["hump"]),"plateau":md(lambda r:r["plateau"]),
         "E_real_blk32":md(lambda r:r["E_med"]),"hit_med":md(lambda r:r["hit"],rows),
         "by_dom":{dm:{"hit":round(float(np.median([r["hit"] for r in rows if r["dom"]==dm])),3),
                       "loopy":round(float(np.mean([r["loopy"] for r in rows if r["dom"]==dm])),2)}
                   for dm in ["wiki","math","code","chat"]},
         "rows":rows}
    return out
if __name__=="__main__":
    res=[]
    res.append(battery("/data/prism_kernels/exp62/gen_layers_qwen.npz",
                       "/data/prism_kernels/exp62/gen_layers_qwen_meta.json","T0_local"))
    for T in ["0.3","0.7","1.0"]:
        p=f"/data/prism_kernels/exp62/gen_T{T}_qwen.npz"
        if os.path.exists(p): res.append(battery(p,p.replace(".npz","_meta.json"),f"T{T}_kaggle"))
        else: print("awaiting",p,flush=True)
    for r in res:
        print(json.dumps({k:r[k] for k in ["tag","n_nondeg","loopy_frac","dip","hump","plateau","E_real_blk32","hit_med","by_dom"]},ensure_ascii=False))
    json.dump(res,open("../results/e64/B4_battery.json","w"),ensure_ascii=False)
    print("saved B4_battery.json")

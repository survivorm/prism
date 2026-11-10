#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76f «CONTEXT MATTERS» (переформулировка пользователя): второй член зависит не от
# текущего состояния, а от какого-то ПРЕДЫДУЩЕГО токена — возможно, последнего
# ЗНАЧИМОГО (сюрприз: низкая p1), не обязательно lag-1.
# Клетка own (E-64 npz). Выравнивание (по e64_gpu.run_doc): vec[j] — состояние ДО
# эмиссии tok[j]; p1[j] — уверенность распределения, выбравшего tok[j]; same[j] —
# совпадение tok[j] с tok[j-1].
# Мера: Δ_t = x_{t+1}−x_t; base-регрессор [x_t]; контекст [x_t, x_{t*}].
#   t*: sig — последний сюрпризный (p1≤q20, обновление ПОСЛЕ использования);
#       same — последний повторыш; minp1 — argmin p1 в прошлом;
#       rnd — случайный шаг (контроль ПАРАМЕТРИЧЕСКОЙ щедрости, та же размерность).
# Индексы ctx для rnd фиксированы сидом на фолд ⇒ сравнимость пар.
# Метрика: LOO по 16 документам; R² по sum SSE / sum TOT; gain(sig−rnd) с
# bootstrap(300) по документам. Пре-рег: gain>0 И CI>0 ⇒ контекст-зависимость
# выше параметрического эквивалента.
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import json
from pathlib import Path
import numpy as np
D=Path("results/e64/tpu"); R=64
LAY=[int(x) for x in os.environ.get("E76F_LAYERS","1,11,23").split(",")]
def ridge_A(Z,T,eps=1e-6):
    G=Z.T@Z; G=G+eps*np.trace(G)/Z.shape[1]*np.eye(Z.shape[1])
    return np.linalg.solve(G,Z.T@T).T   # min ||T − Z Aᵀ||: (ZᵀZ)Aᵀ=ZᵀT
def ctx_idx(kind,f,rng):
    T=f["T"]; idx=np.full(T,-1)
    if kind in ("sig","same"):
        last=-1
        for i in range(T):
            idx[i]=last
            hit=(f["p1"][i]<=f["thr"]) if kind=="sig" else (f["sm"] is not None and bool(f["sm"][i]))
            if hit: last=i
    elif kind=="minp1":
        last=0
        for i in range(1,T):
            idx[i]=last
            if f["p1"][i]<f["p1"][last]: last=i
    elif kind=="rnd":
        idx=rng.integers(0,np.maximum(np.arange(T),2))
    return idx
res_all={}
rng_global=np.random.default_rng(7)
for L in LAY:
    z=np.load(D/"P1_own.npz")
    names=sorted({k[:-3] for k in z.files if k.endswith(f"_L{L}")})
    allrows=np.concatenate([z[f"{nm}_L{L}"].astype(np.float64) for nm in names],0)
    allrows=allrows-allrows.mean(0); allrows/=np.linalg.norm(allrows,axis=1,keepdims=True)
    _,Vv=np.linalg.eigh(np.cov(allrows.T)); B=Vv[:,-R:]
    docs=[]
    for nm in names:
        V=z[f"{nm}_L{L}"].astype(np.float64)
        Vn=V-V.mean(0); Vn=Vn/np.linalg.norm(Vn,axis=1,keepdims=True)
        A=Vn@B
        T=len(A)-1
        f=dict(nm=nm,x=A[:-1],dl=A[1:]-A[:-1],T=T,
               p1=z[f"{nm}_p1"].astype(np.float64)[:T],
               sm=z[f"{nm}_same"][:T] if f"{nm}_same" in z.files else None)
        assert len(f["p1"])==T, (nm,len(f["p1"]),T)
        f["thr"]=np.quantile(f["p1"],0.20)
        docs.append(f)
    N=len(docs)
    def loo_sse(kind):
        sse=[];tot=[]
        for fold in range(N):
            rng=np.random.default_rng(1000+fold)
            Ztr=[];Ttr=[]
            for i,f in enumerate(docs):
                if kind=="base": Z=f["x"]
                else:
                    idx=ctx_idx(kind,f,rng)
                    xc=np.where((idx>=0)[:,None],f["x"][np.maximum(idx,0)],0.0)
                    Z=np.concatenate([f["x"],xc],axis=1)
                if i==fold: Zte,ZT=Z,f["dl"]
                else: Ztr.append(Z);Ttr.append(f["dl"])
            A=ridge_A(np.concatenate(Ztr),np.concatenate(Ttr))
            te=ZT-Zte@A.T
            sse.append(float((te**2).sum()))
            tot.append(float(((ZT-np.concatenate(Ttr).mean(0))**2).sum()))
        return np.array(sse),np.array(tot)
    S={k:loo_sse(k) for k in ["base","sig","minp1","same","rnd"]}
    def R2(k): s,t=S[k]; return 1-s.sum()/t.sum()
    g=S["rnd"][0]-S["sig"][0]; tot=S["sig"][1].sum()
    bs=[g[rng_global.integers(0,N,N)].sum()/tot for _ in range(300)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    rep={k:round(R2(k)-R2("base"),4) for k in ["sig","minp1","same","rnd"]}
    res_all[f"L{L}"]=dict(R2_base=round(R2("base"),4),gain_vs_base=rep,
        gain_sig_minus_rnd=round(float(g.sum()/tot),4),ci95=[round(float(lo),4),round(float(hi),4)])
    print(f"L{L:2d} R2(base) {R2('base'):.3f} | ΔR² vs base: {rep} | "
          f"sig−rnd {g.sum()/tot:+.4f} CI95 [{lo:+.4f},{hi:+.4f}]",flush=True)
(D/"EIGEN_f.json").write_text(json.dumps(res_all,ensure_ascii=False,indent=1))
print("E76f DONE",flush=True)

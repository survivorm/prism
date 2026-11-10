#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76g СЕМАНТИКА второго члена: «импровизация или стереотипия?»
# Внутри own-клетки шаги делятся фазой СЛЕДУЮЩЕГО токена:
#   periodic — lookup (longest повтор. суффикс ≤16, дословно из e64_gpu.lookup_cov)
#              предсказал tok[t];  fresh — lookup промахнулся или предсказал мимо.
# (Позиция: vec-строка t — состояние ДО эмиссии tok[t]; переход t (x_t→x_t+1)
#  относится к фазе tok[t+1].) Плюс контроль-разрез по уверенности: p1 выше/ниже
# медианы (импровизация должна жить в «неуверенных», стереотипия — в «уверенных»).
# Мера: λ₁,λ₂ (сим. спектр одношагового оператора) по фазам, общий базис cov(own),
# bootstrap по документам (200) ⇒ CI на Δλ₂=λ₂(fresh)−λ₂(periodic). Углы НЕ берём:
# фазы неравны по n, на углах это артефакт (УРОК e76b).
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import json
from pathlib import Path
import numpy as np
D=Path("results/e64/tpu"); R=64
LAY=[int(x) for x in os.environ.get("E76G_LAYERS","1,11,23").split(",")]
rng=np.random.default_rng(11)
def lookup_flags(tok):
    """дословная логика lookup_cov, но попозизионно."""
    T=len(tok); hit=np.zeros(T,bool)
    seen={}
    for t in range(1,T):
        pred=None
        for l in range(min(16,t-1),0,-1):
            key=tuple(tok[t-l:t]); p=seen.get(key)
            if p is not None and p+1<len(tok) and p<t-l:
                pred=tok[p+1]; break
        if pred is not None and pred==tok[t]: hit[t]=True
        for l in range(1,min(16,t)+1):
            seen.setdefault(tuple(tok[t-l+1:t+1]),t)
    return hit
def sym_spec(Xs,Ys,k=2):
    Sxy=sum(X.T@Y for X,Y in zip(Xs,Ys)); Sxx=sum(X.T@X for X in Xs)
    M=np.linalg.solve(Sxx.T,Sxy.T).T; w,V=np.linalg.eigh(.5*(M+M.T))
    i=np.argsort(w)[:k]; return w[i]
out={}
for L in LAY:
    z=np.load(D/"P1_own.npz"); sfx=f"_L{L}"
    names=sorted({k[:-len(sfx)] for k in z.files if k.endswith(sfx)})
    allr=np.concatenate([z[f"{n}{sfx}"].astype(np.float64) for n in names],0)
    allr=allr-allr.mean(0); allr/=np.linalg.norm(allr,axis=1,keepdims=True)
    _,Vv=np.linalg.eigh(np.cov(allr.T)); B=Vv[:,-R:]
    per=[]
    for n in names:
        V=z[f"{n}{sfx}"].astype(np.float64); tok=z[f"{n}_tok"]
        p1=z[f"{n}_p1"].astype(np.float64)
        Vn=V-V.mean(0); Vn=Vn/np.linalg.norm(Vn,axis=1,keepdims=True)
        A=Vn@B; Tm=min(A.shape[0],len(tok),len(p1)+1)
        hit=lookup_flags(np.asarray(tok[:Tm]).tolist())
        x=A[:Tm-1]; dl=A[1:Tm]-A[:Tm-1]; fl=hit[1:Tm]; p=p1[:Tm-1]
        med=np.median(p)
        per.append(dict(x=x,dl=dl,hit=fl,p=p,med=med))
    nper=sum(int((d["hit"]).sum()) for d in per); ntot=sum(len(d["hit"]) for d in per)
    def sel(mask_fn):
        Xs=[];Ys=[]
        for d in per:
            m=mask_fn(d)
            if m.sum()>3:
                Xs.append(d["x"][m]); Ys.append(d["x"][m]+d["dl"][m])
        return Xs,Ys
    cuts={"periodic":lambda d:d["hit"],"fresh":lambda d:~d["hit"],
          "conf":lambda d:d["p"]>=d["med"],"surpr":lambda d:d["p"]<d["med"]}
    full={k:sym_spec(*sel(f)) for k,f in cuts.items()}
    # bootstrap Δλ₂ fresh−periodic по документам
    d2=[]
    for _ in range(200):
        ii=rng.integers(0,len(per),len(per))
        sub=[per[i] for i in ii]
        def sel2(mf):
            Xs=[];Ys=[]
            for d in sub:
                m=mf(d)
                if m.sum()>3: Xs.append(d["x"][m]); Ys.append(d["x"][m]+d["dl"][m])
            return Xs,Ys
        lf=sym_spec(*sel2(cuts["fresh"])); lp=sym_spec(*sel2(cuts["periodic"]))
        d2.append(lf[1]-lp[1])
    lo,hi=np.percentile(d2,[2.5,97.5])
    out[f"L{L}"]={k:[round(float(x),3) for x in v] for k,v in full.items()}
    out[f"L{L}"]["d2_fresh_minus_periodic"]=round(float(np.median(d2)),3)
    out[f"L{L}"]["ci95"]=[round(float(lo),3),round(float(hi),3)]
    out[f"L{L}"]["frac_periodic"]=round(nper/ntot,3)
    print(f"L{L:2d} λ(1,2): periodic {full['periodic'].round(2)} fresh {full['fresh'].round(2)} "
          f"| conf {full['conf'].round(2)} surpr {full['surpr'].round(2)} | periodic доля {nper/ntot:.2f}",flush=True)
    print(f"     Δλ₂ fresh−periodic {np.median(d2):+.3f} CI95 [{lo:+.3f},{hi:+.3f}]",flush=True)
(D/"EIGEN_g.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E76g DONE",flush=True)

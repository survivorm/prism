#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-80b NULL для e76k-сцепки: |ρ(e_k,surp)| с CI «по парам» не есть значимость —
нужен нуль, сохраняющий автокорреляцию сюрприза. own-сторона целиком на CPU
(p1 в P1_own.npz). Нули: circular shift ≥8 (сохраняет ACF сюрпрайза) и shuffle
(убивает всё). 200 повторов, репорт: observed vs p95 нуля, лаги 0..2, e1/e2.
cor-сторона требует teacher-forced p1 (GPU) — сюда не входит, см. EIGEN_k позже.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e80"; OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(8080); R=2; L=23; NB=200
z_o=np.load(D/"P1_own.npz"); z_c=np.load(D/"P2_corpus.npz"); sfx=f"_L{L}"
def rows(z):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
Uo=rows(z_o); Uc=rows(z_c); assert len(Uo)==16==len(Uc)
Call=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(Call);B=V[:,-R:]
Xs=[];Ys=[]
for U in Uo+Uc:
    A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x in Xs)
M=np.linalg.solve(Sxx.T,Sxy.T).T;w,Vv=np.linalg.eigh(.5*(M+M.T))
pl=Vv[:,np.argsort(w)[:2]]
Eo=[]
for U in Uo:
    A=U@B; Dl=A[1:]-A[:-1]
    n2=np.sum(Dl*Dl,axis=1)+1e-12
    Eo.append(np.stack([(Dl@pl[:,k])**2/n2 for k in range(2)],1))
skeys=[k for k in sorted(z_o.files) if k.endswith("_p1") and k.startswith("own")]
surp=[-np.log(np.clip(z_o[k].astype(np.float64),1e-6,1)) for k in skeys]
assert len(surp)==16, (len(surp),skeys[:3])
def lagc(E,surp,k,ch):
    xs=[];ys=[]
    for e,s in zip(E,surp):
        Tm=min(len(e),len(s)); e=e[:Tm];s=s[:Tm]
        idx=np.arange(8,Tm); idx=idx[idx>=k] if k>0 else idx
        xs.append(e[idx,ch]); ys.append(s[(idx-k) if k>0 else idx])
    x=np.concatenate(xs);y=np.concatenate(ys)
    return float(np.corrcoef(x,y)[0,1])
def shifted(s,kind):
    T=len(s)
    if kind=="shift":
        k=int(RNG.integers(8,max(9,T//2))); return np.roll(s,k)
    p=RNG.permutation(T); return s[p]
out={}
for ch,nm in ((0,"e1"),(1,"e2")):
    for k in range(3):
        obs=lagc(Eo,surp,k,ch)
        for kind in ("shift","shuffle"):
            null=[]
            for _ in range(NB):
                sn=[shifted(s,kind) for s in surp]
                null.append(lagc(Eo,sn,k,ch))
            hi=np.percentile(null,97.5); lo=np.percentile(null,2.5)
            out[f"{nm}_lag{k}_{kind}"]=dict(obs=round(obs,4),
                null_p95=[round(float(lo),4),round(float(hi),4)],
                signif=bool(obs<lo or obs>hi))
            print(f"{nm} lag{k} {kind:8s}: obs {obs:+.3f} null[{lo:+.3f},{hi:+.3f}] "
                  +("SIGNIF" if (obs<lo or obs>hi) else "—"),flush=True)
(OUT/"NULL_e76k_own.json").write_text(json.dumps(out,indent=1,ensure_ascii=False))
print("E80B DONE",flush=True)

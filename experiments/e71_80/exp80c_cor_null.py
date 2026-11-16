#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-80c NULL, корпусная сторона: тот же протокол, что e80b, для cor.
Входы: results/e64/tpu/cor_p1.npz (пишет exp76k, запуск после e79) + P2_corpus.npz.
Чистый CPU. Нули: circular shift >=8 и shuffle, 200 повторов, e1/e2, лаги 0..2.
Вопрос: переживает ли сдвиг-нуль связь cor_e2~surp (+0.096 в EIGEN_k), которая в
e76k оказалась СИЛЬНЕЕ own — если да, «поток-механизм» двусторонне реален и различие
own/cor только в знаке e1; если нет — корпусная «сильность» была артефактом.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e80"
RNG=np.random.default_rng(8081); R=2; L=23; NB=200
zp=np.load(D/"cor_p1.npz"); zc=np.load(D/"P2_corpus.npz"); sfx=f"_L{L}"
def rows(z):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
Uc=rows(zc); zo=np.load(D/"P1_own.npz"); Uo=rows(zo)
assert len(Uc)==16==len(Uo)
pkeys=sorted(zp.files)
surp=[-np.log(np.clip(zp[k].astype(np.float64),1e-6,1)) for k in pkeys]
assert len(surp)==16
# парность U↔p1: имена U 'cor_w{i}_L23' ↔ p1 'cor_{i:02d}' — сверяем индексы явно
idx_U=[int(k.split("_w")[1].split("_")[0]) for k in sorted([f for f in zc.files if f.endswith(sfx)])]
idx_p=[int(k.split("_")[1]) for k in pkeys]
assert idx_U==idx_p, (idx_U[:5],idx_p[:5])
Call=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(Call);B=V[:,-R:]
Xs=[];Ys=[]
for U in Uo+Uc:
    A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x in Xs)
M=np.linalg.solve(Sxx.T,Sxy.T).T;w,Vv=np.linalg.eigh(.5*(M+M.T))
pl=Vv[:,np.argsort(w)[:2]]
Ec=[]
for U in Uc:
    A=U@B; Dl=A[1:]-A[:-1]
    n2=np.sum(Dl*Dl,axis=1)+1e-12
    Ec.append(np.stack([(Dl@pl[:,k])**2/n2 for k in range(2)],1))
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
    return s[RNG.permutation(T)]
out={}
for ch,nm in ((0,"e1"),(1,"e2")):
    for k in range(3):
        obs=lagc(Ec,surp,k,ch)
        for kind in ("shift","shuffle"):
            null=[]
            for _ in range(NB):
                sn=[shifted(s,kind) for s in surp]
                null.append(lagc(Ec,sn,k,ch))
            lo,hi=np.percentile(null,2.5),np.percentile(null,97.5)
            out[f"{nm}_lag{k}_{kind}"]=dict(obs=round(obs,4),
                null_p95=[round(float(lo),4),round(float(hi),4)],
                signif=bool(obs<lo or obs>hi))
            print(f"cor {nm} lag{k} {kind:8s}: obs {obs:+.3f} null[{lo:+.3f},{hi:+.3f}] "
                  +("SIGNIF" if (obs<lo or obs>hi) else "—"),flush=True)
(OUT/"NULL_e76k_cor.json").write_text(json.dumps(out,indent=1,ensure_ascii=False))
print("E80C DONE",flush=True)

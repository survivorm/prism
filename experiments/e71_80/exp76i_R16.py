#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76i РОБАСТНОСТЬ НА СЖАТОМ БАЗИСЕ (R=16, пре-рег E-76h): проверяем, что три
# вывода серии держатся без 64-м базиса:
#  (1) Δλ₂(own−corpus)<0 со всех слоёв (bootstrap CI) — «слагаемое везде»;
#  (2) U-матрица |u_i·u_j| 2×2 — общая λ₁;
#  (3) fresh−periodic Δλ₂ внутри own (семантика «импровизация») — CI сжатым базисом.
# Пре-рег: вывод робастен, если (1) знак+CI<0 на ≥4/5 слоёв, (2) диагональ U≥0.6 в
# L1/L11, (3) знак fresh−periodic совпадёт с R=64 (−) и CI сузится.Null не нужен —
# сравнения направленные внутри одной размерности.
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import json
from pathlib import Path
import numpy as np
D=Path("results/e64/tpu"); R=16; LAY=[1,5,11,17,23]
rng=np.random.default_rng(77)
def rows(path,layer,names=False):
    z=np.load(path); o=[];nm=[];sfx=f"_L{layer}"
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12));nm.append(k[:-len(sfx)])
    return (o,nm) if names else o
def XY(U,B): A=U@B; return A[:-1],A[1:]
def spec(Xs,Ys):
    Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x,y in zip(Xs,Ys))
    M=np.linalg.solve(Sxx.T,Sxy.T).T;w,V=np.linalg.eigh(.5*(M+M.T))
    i=np.argsort(w)[:2];return V[:,i],w[i]
def lookup_flags(tok):
    T=len(tok);hit=np.zeros(T,bool);seen={}
    for t in range(1,T):
        pred=None
        for l in range(min(16,t-1),0,-1):
            key=tuple(tok[t-l:t]);p=seen.get(key)
            if p is not None and p+1<len(tok) and p<t-l: pred=tok[p+1];break
        if pred is not None and pred==tok[t]: hit[t]=True
        for l in range(1,min(16,t)+1): seen.setdefault(tuple(tok[t-l+1:t+1]),t)
    return hit
out={}
for L in LAY:
    Uc=rows(D/"P2_corpus.npz",L);no=rows(D/"P1_own.npz",L,True)
    Uo,no=rows(D/"P1_own.npz",L,True)
    Call=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(Call);B=V[:,-R:]
    Xc=[XY(u,B)[0] for u in Uc];Yc=[XY(u,B)[1] for u in Uc]
    Xo=[XY(u,B)[0] for u in Uo];Yo=[XY(u,B)[1] for u in Uo]
    Pc,lc=spec(Xc,Yc);Po,lo=spec(Xo,Yo)
    Umat=np.abs(Pc.T@Po)
    dl=[]
    for k in range(2):
        bs=[]
        for _ in range(300):
            ic=rng.integers(0,16,16);io=rng.integers(0,16,16)
            _,a=spec([Xc[i] for i in ic],[Yc[i] for i in ic])
            _,b=spec([Xo[i] for i in io],[Yo[i] for i in io])
            bs.append(b[k]-a[k])
        dl.append([round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)])
    # fresh/periodic в сжатом базисе
    per=[]
    for nm in no:
        V=z_=np.load(D/"P1_own.npz");V=z_[f"{nm}_L{L}"].astype(np.float64);tok=z_[f"{nm}_tok"]
        Vn=V-V.mean(0);Vn/=np.linalg.norm(Vn,axis=1,keepdims=True)
        A=Vn@B;Tm=min(A.shape[0],len(tok));x=A[:Tm-1];d=A[1:Tm]-A[:Tm-1]
        hit=lookup_flags(np.asarray(tok[:Tm]).tolist())[1:Tm]
        per.append((x,d,hit))
    def sp(m):
        Xs=[];Ys=[]
        for x,d,h in per:
            mm=m(h)
            if mm.sum()>3: Xs.append(x[mm]);Ys.append(x[mm]+d[mm])
        return spec(Xs,Ys)[1]
    lf=sp(lambda h:~h);lp=sp(lambda h:h)
    bd=[]
    for _ in range(200):
        ii=rng.integers(0,len(per),len(per));sub=[per[i] for i in ii]
        def sp2(m):
            Xs=[];Ys=[]
            for x,d,h in sub:
                mm=m(h)
                if mm.sum()>3:Xs.append(x[mm]);Ys.append(x[mm]+d[mm])
            return spec(Xs,Ys)[1]
        bd.append(sp2(lambda h:~h)[1]-sp2(lambda h:h)[1])
    lo95,hi95=np.percentile(bd,[2.5,97.5])
    out[f"L{L}"]=dict(lam_c=[round(float(x),2) for x in lc],lam_o=[round(float(x),2) for x in lo],
        dlam=[round(float(lo[k]-lc[k]),3) for k in range(2)],ci=dl,
        U=[[round(float(Umat[i,j]),2) for j in range(2)] for i in range(2)],
        d2_fresh_per=round(float(np.median(bd)),3),ci2=[round(float(lo95),3),round(float(hi95),3)])
    print(f"L{L:2d} Δλ {out[f'L{L}']['dlam']} CI {dl} | Udiag {out[f'L{L}']['U'][0][0]},{out[f'L{L}']['U'][1][1]} | fresh−per {out[f'L{L}']['d2_fresh_per']:+.2f} CI [{lo95:+.2f},{hi95:+.2f}]",flush=True)
(D/"EIGEN_i.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E76i DONE")

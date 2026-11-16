#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-80 RING: «затухающий гул» e1? Автокорреляции ЗНАКОВОЙ проекции шагов Δ на
плоскость u1/u2 (общий базис own+cor, как в e76k), own vs corpus, по лагам 1..6.
Пре-рег (до запуска):
 1)Null-пол: ACF1 сглаженных приращений заведомо отрицателен (эффектив сглаживания).
   Значим = НЕ уровень ACF1, а ПРОФИЛЬ: нуль-пересечение ACF(k) по k и возврат
   в плюс (кольцевание) vs монотонное затухание (просто шум с трендом).
 2)«Гул» = own-профиль с пересечением нуля на k*=2..4 и медианой ACF(k*)>+0.05
   при ACF(1)<0. Если профили own и cor совпадают (различия внутри бутстрап-CI) —
   «собственного резонанса» нет, e1-эффект e76k = общий отклик потока.
 3)Слои: считаем L11 и L23 (L1/L5 внеInterest — носители, не readout).
Чистый CPU по существующим npz; новых forward нет.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e80"; OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(80)

def rows(z,sfx):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
def plane(Uo,Uc,R=2):
    All=np.concatenate(Uc+Uo,0)
    C=np.cov(All.T);_,V=np.linalg.eigh(C);B=V[:,-R:]
    Xs=[];Ys=[]
    for U in Uc+Uo:
        A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
    Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x in Xs)
    M=np.linalg.solve(Sxx.T,Sxy.T).T;w,Vv=np.linalg.eigh(.5*(M+M.T))
    return B,Vv[:,np.argsort(w)[:2]]
def signed_acf(Us,plane,kmax=6):
    """ACF знаковой проекции p(t)=unit(Dl)@e_k по лагам; по документам отдельно."""
    out=[]
    for U in Us:
        A=U@plane@np.eye(plane.shape[1]) if False else None
    for U in Us:
        Bm=None
    return out
# --- аккуратно: две плоскости (e1,e2) от одного B ---
def series(Us,B,pl):
    per=[]
    for U in Us:
        A=U@B; Dl=A[1:]-A[:-1]
        n=np.linalg.norm(Dl,axis=1,keepdims=True)+1e-12
        per.append(np.stack([ (Dl/n)@pl[:,k] for k in range(pl.shape[1])],1))  # [T,2]
    return per
def acf_of(p,kmax):
    p=p-p.mean(); v=np.dot(p,p)+1e-12
    return [float(np.dot(p[:-k],p[k:])/v) for k in range(1,kmax+1)]
def boot_ci(vals,Bn=4000):
    a=np.array(vals); idx=RNG.integers(0,len(a),(Bn,len(a)))
    m=np.median(a[idx],axis=1); return float(np.median(a)),[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]

res={}
for L in (11,23):
    zo=np.load(D/f"P1_own.npz"); zc=np.load(D/f"P2_corpus.npz"); sfx=f"_L{L}"
    Uo=rows(zo,sfx); Uc=rows(zc,sfx)
    assert len(Uo)==16 and len(Uc)==16, (L,len(Uo),len(Uc))
    B,pl=plane(Uo,Uc)
    so=series(Uo,B,pl); sc=series(Uc,B,pl)
    res[f"L{L}"]={}
    for k,tag in ((0,"e1"),(1,"e2")):
        Ao=[acf_of(s[:,k]) for s in so]; Ac=[acf_of(s[:,k]) for s in sc]
        row={}
        for lag in range(6):
            mo,co=boot_ci([a[lag] for a in Ao]); mc,cc=boot_ci([a[lag] for a in Ac])
            dif=[b[lag]-c[lag] for b,c in zip(Ao,Ac)]
            md,cd=boot_ci(dif)
            row[f"lag{lag+1}"]=dict(own=[round(mo,3),[round(x,3) for x in co]],
                                    cor=[round(mc,3),[round(x,3) for x in cc]],
                                    dif=[round(md,3),[round(x,3) for x in cd]])
        res[f"L{L}"][tag]=row
        print(f"L{L} {tag}: own "+" ".join(f"{row[f'lag{i+1}']['own'][0]:+.2f}" for i in range(6)),flush=True)
        print(f"        cor "+" ".join(f"{row[f'lag{i+1}']['cor'][0]:+.2f}" for i in range(6)),flush=True)
        print(f"        d± "+" ".join(f"{row[f'lag{i+1}']['dif'][0]:+.2f}"+("*" if (row[f'lag{i+1}']['dif'][1][0]>0 or row[f'lag{i+1}']['dif'][1][1]<0) else " ") for i in range(6)),flush=True)
# вердикт-гейт: пересечение нуля у own e1
verd={}
for L in (11,23):
    r=res[f"L{L}"]["e1"]; own=[r[f"lag{k}"]["own"][0] for k in range(1,7)]
    kx=next((k for k in range(2,7) if own[k-1]>0 and own[k-2]<0),None)
    amp=r[f"lag{kx}"]["own"] if kx else None
    verd[f"L{L}"]=dict(zero_cross_k=kx,amp_at_cross=amp,
                       ring=bool(kx is not None and amp is not None and amp[0]>0.05))
res["verdict"]=verd
(OUT/"RING.json").write_text(json.dumps(res,ensure_ascii=False,indent=1,allow_nan=False))
print("E80 DONE",json.dumps(verdict if False else verd,ensure_ascii=False),flush=True)

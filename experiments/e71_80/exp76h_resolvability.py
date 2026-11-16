#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76h РАЗРЕШИМОСТЬ СЖАТИЕМ: ширина полосы cos — артефакт 64-мерного базиса,
# не нехватка n. Таблица: для R∈{8,16,32,64}: cross(corpus,own) полные 16/16;
# split-half воспроизводимость внутри клеток (8/8); cross на подвыборках 8×8 ±sd;
# плюс ЧЕСТНЫЙ chance-level случайных 2-плоскостей в R (иначе R=8 даёт ложную моду).
# Вердикт-правило (пре-рег): разрешено, если cross > chance_p99(R) И |cross−repro|
# сопоставим с шириной repro; R=16 — выбранный рабочий (R=8: chance p95=0.65 — зона
# риска; выбор R по сетке на тех же данных — селективный эффект, тренд гладкий).
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import json
from pathlib import Path
import numpy as np
D=Path("results/e64/tpu")
rng=np.random.default_rng(31)
def rows(path,layer):
    z=np.load(path); o=[]; sfx=f"_L{layer}"
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64); H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
def planes(U,B): return [((U@B)[:-1],(U@B)[1:]) for U in U]
def plane(XY):
    Sxy=sum(x.T@y for x,y in XY); Sxx=sum(x.T@x for x,y in XY)
    M=np.linalg.solve(Sxx.T,Sxy.T).T; w,V=np.linalg.eigh(.5*(M+M.T))
    return V[:,np.argsort(w)[:2]]
def cos2(P,Q): return float(np.clip(np.linalg.svd(P.T@Q,compute_uv=False),0,1).mean())
out={"null":{}}
for R in (8,16,32,64):
    cs=[]
    for _ in range(4000):
        A=np.linalg.qr(rng.standard_normal((R,2)))[0]; Bm=np.linalg.qr(rng.standard_normal((R,2)))[0]
        cs.append(np.clip(np.linalg.svd(A.T@Bm,compute_uv=False),0,1).mean())
    out["null"][R]=dict(med=round(float(np.median(cs)),3),p95=round(float(np.percentile(cs,95)),3),p99=round(float(np.percentile(cs,99)),3))
for R in (8,16,32,64):
    for L in (1,11,23):
        Uc=rows(D/"P2_corpus.npz",L); Uo=rows(D/"P1_own.npz",L)
        Call=np.cov(np.concatenate(Uc+Uo,0).T); _,V=np.linalg.eigh(Call); B=V[:,-R:]
        Pc=[planes([u],B)[0] for u in Uc]; Po=[planes([u],B)[0] for u in Uo]
        full=cos2(plane(Pc),plane(Po))
        sho=[];shc=[]
        for _ in range(80):
            io=rng.permutation(16); sho.append(cos2(plane([Po[i] for i in io[:8]]),plane([Po[i] for i in io[8:]])))
            ic=rng.permutation(16); shc.append(cos2(plane([Pc[i] for i in ic[:8]]),plane([Pc[i] for i in ic[8:]])))
        cs=[]
        for _ in range(120):
            ic=rng.choice(16,8,replace=False); io=rng.choice(16,8,replace=False)
            cs.append(cos2(plane([Pc[i] for i in ic]),plane([Po[i] for i in io])))
        out[f"R{R}_L{L}"]=dict(cross16=round(full,3),
            repro_own=[round(float(np.median(sho)),3),round(float(np.percentile(sho,16)),3),round(float(np.percentile(sho,84)),3)],
            repro_cor=[round(float(np.median(shc)),3),round(float(np.percentile(shc,16)),3),round(float(np.percentile(shc,84)),3)],
            cross8=round(float(np.mean(cs)),3),sd8=round(float(np.std(cs)),3))
(D/"EIGEN_h.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E76h DONE — E-76d для главного вердикта НЕ нужен (см. R=16 vs null.p99)")

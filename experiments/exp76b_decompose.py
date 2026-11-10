#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76b РАЗЛОЖЕНИЕ «0.70»: одна ли мода общая, или средний cos 2-плоскостей размыт
# вторым, петле-собственным низкочастотным модом (гипотеза пользователя)?
# Меры (все на общем базисе corpus∪own, слои [1,11,23]):
#  c1 = |u1_corpus · u1_own|;  c2 = |u2_corpus · u2_own| (попарно по индексу λ)
#  P(cu1→plane_own), P(ou1→plane_corpus) — синусы угла; если corpus λ₁ лежит в петлевой
#    плоскости, а петлевой u2 ортогонален corpus-плоскости ⇒ «одна общая + добавок у петли»
#  bootstrap(доки, 300): CI для cos углов 2-плоскостей и для c1 — узость решения.
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import ast, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; R=64
def unit_rows(path, layer):
    z=np.load(path); Us=[]
    for k in sorted(z.files):
        if k.endswith(f"_L{layer}"):
            H=z[k].astype(np.float64); H-=H.mean(0)
            Us.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return Us
def planes(rows, B):
    Xs=[];Ys=[]
    for U in rows:
        A=U@B; Xs.append(A[:-1]); Ys.append(A[1:])
    return Xs,Ys
def lam_plane(Xs,Ys,k=2):
    Rr=Xs[0].shape[1]
    Sxy=sum(X.T@Y for X,Y in zip(Xs,Ys)); Sxx=sum(X.T@X for X in Xs)
    M=np.linalg.solve(Sxx.T,Sxy.T).T; w,V=np.linalg.eigh(.5*(M+M.T))
    i=np.argsort(w)[:k]
    return V[:,i], w[i]
def canon(a,b):
    return np.clip(np.linalg.svd(a.T@b,compute_uv=False),0,1)
def proj_in(u,plane):
    s=np.linalg.svd(plane.T@u.reshape(-1,1),compute_uv=False)
    return float(np.clip(s[0],0,1))
out={}
rng=np.random.default_rng(0)
for L in (1,11,23):
    Uc=unit_rows(D/"P2_corpus.npz",L); Uo=unit_rows(D/"P1_own.npz",L)
    Call=np.cov(np.concatenate(Uc+Uo,0).T); _,V=np.linalg.eigh(Call); B=V[:,-R:]
    Xc,Yc=planes(Uc,B); Xo,Yo=planes(Uo,B)
    Pc,lc=lam_plane(Xc,Yc); Po,lo=lam_plane(Xo,Yo)
    c=canon(Pc,Po)
    # по-направленные cos (макс по собственному числу)
    c1=abs(float(Pc[:,0]@Po[:,0])); c2=abs(float(Pc[:,1]@Po[:,1]))
    p_cu1=proj_in(Pc[:,0],Po); p_ou1=proj_in(Po[:,0],Pc)
    p_ou2=proj_in(Po[:,1],Pc)   # петлевой второй мод в corpus-плоскости?
    # bootstrap
    bc=[];bc1=[]
    n_c,n_o=len(Xc),len(Xo)
    for _ in range(300):
        ic=rng.integers(0,n_c,n_c); io=rng.integers(0,n_o,n_o)
        Pb,_=lam_plane([Xc[i] for i in ic],[Yc[i] for i in ic])
        Qb,_=lam_plane([Xo[i] for i in io],[Yo[i] for i in io])
        bc.append(canon(Pb,Qb).mean()); bc1.append(abs(float(Pb[:,0]@Qb[:,0])))
    lo95,hi95=np.percentile(bc,[2.5,97.5]); l1,h1=np.percentile(bc1,[2.5,97.5])
    out[f"L{L}"]=dict(cos2_mean=round(float(c.mean()),3),canon=[round(float(x),3) for x in c],
        c1=c1 and round(c1,3), c2=round(c2,3),
        proj_cu1_in_own=round(p_cu1,3), proj_ou1_in_cor=round(p_ou1,3), proj_ou2_in_cor=round(p_ou2,3),
        lam_c=[round(float(x),2) for x in lc], lam_o=[round(float(x),2) for x in lo],
        cos2_ci95=[round(float(lo95),3),round(float(hi95),3)], c1_ci95=[round(float(l1),3),round(float(h1),3)])
    print(f"L{L}: cos2 {out[f'L{L}']['cos2_mean']} CI {out[f'L{L}']['cos2_ci95']} | c1 {out[f'L{L}']['c1']} CI {out[f'L{L}']['c1_ci95']} c2 {out[f'L{L}']['c2']}",flush=True)
    print(f"     corpusλ1 в петл.плоск {p_cu1:.2f} | петлевой u2 в corpus-плоск {p_ou2:.2f} | λc {out[f'L{L}']['lam_c']} λo {out[f'L{L}']['lam_o']}",flush=True)
(D/"EIGENb.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E76b DONE",flush=True)

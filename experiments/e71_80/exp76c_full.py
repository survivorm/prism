#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76c FULL: разделение «режим vs погрешность измерения» для каждой собственной
# компоненты k=1,2, без предположения о «переключении».
#  (а) Δλ_k = λ_k(own) − λ_k(corpus) с bootstrap-CI (для eigenvalues bootstrap
#      корректен; для углов подпространств — нет, там split-half, см. exp76b/этот файл);
#  (б) профиль по ВСЕМ слоям E-64 [1,5,11,17,23]: рост vs скачок;
#  (в) U_ij=|u_i^cor·u_j^own| — матрица 2×2 (индексный свап/микширование);
#  (г) энергия компоненты на документе e_k(d)=mean_t <Δh_t,u_k>²/||Δh_t||² и
#      Спирмен с dip этого же документа (meta E-64) — генератор гипотез «за что отвечает»;
#  (д) ридж Sxx+ε·tr/R·I: устойчивость плоскости к погрешности second moment (n=16).
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import json
from pathlib import Path
import numpy as np
D=Path("results/e64/tpu"); R=64; K=2; LAYERS=[1,5,11,17,23]
rng=np.random.default_rng(7)
def unit_rows(path, layer):
    """rows + ИМЕНА в одном порядке (sorted лексикографический: w0,w1,w10,... — числа
tолько из имён, иначе dip'ы склеятся не со своими рядами)."""
    z=np.load(path); Us=[]; nm=[]
    for k in sorted(z.files):
        if k.endswith(f"_L{layer}"):
            H=z[k].astype(np.float64); H-=H.mean(0)
            Us.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12)); nm.append(k[: -len(f"_L{layer}")])
    return Us,nm
def planes(rows,B):
    Xs=[];Ys=[]
    for U in rows:
        A=U@B; Xs.append(A[:-1]); Ys.append(A[1:])
    return Xs,Ys
def lam_plane(Xs,Ys,ridge=0.0):
    Rr=Xs[0].shape[1]
    Sxy=sum(X.T@Y for X,Y in zip(Xs,Ys)); Sxx=sum(X.T@X for X in Xs)
    if ridge: Sxx=Sxx+ridge*np.trace(Sxx)/Rr*np.eye(Rr)
    M=np.linalg.solve(Sxx.T,Sxy.T).T; w,V=np.linalg.eigh(.5*(M+M.T))
    i=np.argsort(w)[:K]; return V[:,i], w[i]
def spearman(x,y):
    def rk(a):
        o=np.argsort(a); r=np.empty(len(a)); r[o]=np.arange(len(a)); return r
    rx,ry=rk(x),rk(y); s=np.std(rx)*np.std(ry)
    return float(np.mean((rx-rx.mean())*(ry-ry.mean()))/(s*s+1e-12))
meta_c=json.load(open(D/"P1_own_meta.json")); meta_p=json.load(open(D/"P2_corpus_meta.json"))
out={}
for L in LAYERS:
    Uc,nc=unit_rows(D/"P2_corpus.npz",L); Uo,no=unit_rows(D/"P1_own.npz",L)
    Call=np.cov(np.concatenate(Uc+Uo,0).T); _,V=np.linalg.eigh(Call); B=V[:,-R:]
    Xc,Yc=planes(Uc,B); Xo,Yo=planes(Uo,B)
    Pc,lc=lam_plane(Xc,Yc); Po,lo=lam_plane(Xo,Yo)
    Pc_,lc_=lam_plane(Xc,Yc,ridge=1e-3); Po_,lo_=lam_plane(Xo,Yo,ridge=1e-3)
    Umat=np.abs(Pc.T@Po)
    # bootstrap Δλ_k (доки с повторами, 300)
    d=[]
    for k in range(K):
        bs=[]
        for _ in range(300):
            ic=rng.integers(0,len(Xc),len(Xc)); io=rng.integers(0,len(Xo),len(Xo))
            _,lc2=lam_plane([Xc[i] for i in ic],[Yc[i] for i in ic])
            _,lo2=lam_plane([Xo[i] for i in io],[Yo[i] for i in io])
            bs.append(lo2[k]-lc2[k])
        d.append([round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)])
    # энергии компонент на документе и связь с dip
    def energies(rows,B,plane):
        e=[]
        for U in rows:
            A=U@B; Dl=A[1:]-A[:-1]
            n2=np.sum(Dl*Dl,axis=1)+1e-12
            c=np.stack([np.mean((Dl@plane[:,k])**2/n2) for k in range(K)])
            e.append(c)
        return np.array(e)   # [N,K]
    Bc=B; ec=energies(Uc,Bc,Pc); eo=energies(Uo,Bc,Po)
    dip_c=np.array([meta_p[n]["curves"][str(L)]["dip"] for n in nc])
    dip_o=np.array([meta_c[n]["curves"][str(L)]["dip"] for n in no])
    rho={"cor":[round(spearman(ec[:,k],dip_c),3) for k in range(K)],
         "own":[round(spearman(eo[:,k],dip_o),3) for k in range(K)]}
    out[f"L{L}"]=dict(lam_c=[round(float(x),3) for x in lc],lam_o=[round(float(x),3) for x in lo],
        dlam=[round(float(lo[k]-lc[k]),3) for k in range(K)], ci95_dlam=d,
        ridge_shift=[round(float(x),3) for x in (lc_-lc)], ridge_shift_o=[round(float(x),3) for x in (lo_-lo)],
        U=[[round(float(Umat[i,j]),2) for j in range(K)] for i in range(K)],
        rho_dip=rho)
    print(f"L{L:2d} λc={[round(float(x),2) for x in lc]} λo={[round(float(x),2) for x in lo]} "
          f"Δλ={[round(float(lo[k]-lc[k]),2) for k in range(K)]} CI {d}",flush=True)
    print(f"     U={Umat.round(2).tolist()} | ρ(e_k,dip) cor {rho['cor']} own {rho['own']} | ridge Δλ {out[f'L{L}']['ridge_shift']}",flush=True)
(D/"EIGENc.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E76c DONE",flush=True)

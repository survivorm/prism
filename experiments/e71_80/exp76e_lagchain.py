#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76e «БЕЗУМНАЯ ИДЕЯ» (пользователь): второй мод = след волны ПРЕДЫДУЩЕГО токена
# (другой фазы), возможно с хвостом n-2, n-3...
# Формализация: пусть Δ_t — приращение (шаг) в проекции на общий базис. Лаг-оператор
# L_k = argmin ||Δ_{t+k} − L Δ_t|| (k=1..4), т.е. собственная ЭВОЛЮЦИЯ шагов, а не один
# переход. Предсказания версии «цепочка фаз»:
#  (1) u2 ≈ норм. L₁u₁  (второй мод = образ первого за шаг) — мера c2|Mu1;
#  (2) спектр L_k в 2D-плоскости ≈ (−1)^k·ρ^k, т.е. собственные L₁ ≈ −ρ (period-2
#      с затуханием), L₂ ≈ +ρ² — знак чередующийся;
#  (3) «волна n-2»: |u1·v₁(L₂)| велик (первый мод лаг-2 совпадает с u1 — та же волна
#      вернулась через шаг); |u2·v₁(L₂)| велик ⇒ цепочка u1→u2→u1 (фаза крутится в паре).
# Всё офлайн на npz E-64, обе клетки. Если c2|Mu1 мала и знак не чередуется — идея мертва.
import os; os.environ["OPENBLAS_NUM_THREADS"]="4"
import json
from pathlib import Path
import numpy as np
D=Path("results/e64/tpu"); R=64; K=2
LAY=[int(x) for x in os.environ.get("E76E_LAYERS","1,11,23").split(",")]
LAGS=[1,2,3,4]
def unit_rows(path, layer):
    z=np.load(path); Us=[]; nm=[]
    for k in sorted(z.files):
        if k.endswith(f"_L{layer}"):
            H=z[k].astype(np.float64); H-=H.mean(0)
            Us.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12)); nm.append(k)
    return Us
def proj_rows(rows,B):
    return [ (U@B) for U in rows ]
def op(S_target_pred, S_pred):  # предсказатель a: min ||y − a x||
    Rr=S_pred.shape[0]
    return np.linalg.solve(S_pred.T+1e-9*np.eye(Rr),S_target_pred.T).T
def sym_eig(M,k=2):
    w,V=np.linalg.eigh(.5*(M+M.T)); i=np.argsort(w)[:k]; return V[:,i], w[i]
out={}
for L in LAY:
    Uc=unit_rows(D/"P2_corpus.npz",L); Uo=unit_rows(D/"P1_own.npz",L)
    Call=np.cov(np.concatenate(Uc+Uo,0).T); _,V=np.linalg.eigh(Call); B=V[:,-R:]
    res={}
    for cell,rows in (("cor",Uc),("own",Uo)):
        A=proj_rows(rows,B)
        Xs=[a[:-1] for a in A]; Ys=[a[1:] for a in A]; Ds=[a[1:]-a[:-1] for a in A]
        Sdd=sum(d.T@d for d in Ds)
        M1=op(sum(x.T@y for x,y in zip(Xs,Ys)),sum(x.T@x for x in Xs))
        P1,lc=sym_eig(M1)                      # мода λ₁ и второй мод лаг-1
        # моды лаг-k операторов на Δ
        v={}; 
        for k in LAGS:
            Sxy=sum(d[:-k].T@d[k:] for d in Ds)
            Lk=op(Sxy,Sdd); wk,Vk=np.linalg.eigh(.5*(Lk+Lk.T))
            i=np.argsort(wk)[:K]; v[k]=(Vk[:,i],wk[i],sym_eig(Lk,K)[1])
        # (1) u2 vs образ u1
        mu1=M1@P1[:,0]; mu1/=np.linalg.norm(mu1)+1e-12
        c2Mu1=float(abs(P1[:,1]@mu1))
        # инвариантность плоскости: остаток M1 P за пределами span(P)
        resq=M1@P1-P1@(P1.T@(M1@P1)); invar=float(np.linalg.norm(resq)/np.linalg.norm(M1@P1))
        # (2) знак: собств. L₁ (полные, не только сим.) — берём вещ. часть eig
        L1=op(sum(d[:-1].T@d[1:] for d in Ds),Sdd)
        e1=np.sort(np.real(np.linalg.eigvals(L1)))[:K]
        L2=op(sum(d[:-2].T@d[2:] for d in Ds),Sdd)
        e2=np.sort(np.real(np.linalg.eigvals(L2)))[:K]
        # (3) возврата мод
        ov12=float(abs(P1[:,0]@v[2][0][:,0])); ov22=float(abs(P1[:,1]@v[2][0][:,0]))
        ov11=float(abs(P1[:,0]@v[1][0][:,0])); ov21=float(abs(P1[:,1]@v[1][0][:,0]))
        res[cell]=dict(lam1=[round(float(x),3) for x in lc],
            c2_Mu1=round(c2Mu1,3), invar_resid=round(invar,3),
            eigL1=[round(float(x),3) for x in e1], eigL2=[round(float(x),3) for x in e2],
            lag_eigs={str(k):[round(float(x),3) for x in v[k][2]] for k in LAGS},
            ov_u1_v1={ "L1":round(ov11,3),"L2":round(ov12,3)},
            ov_u2_v1={ "L1":round(ov21,3),"L2":round(ov22,3)})
    out[f"L{L}"]=res
    for cell in res:
        r=res[cell]
        print(f"L{L:2d} {cell}: u2↔M·u1 {r['c2_Mu1']:+.3f} | инвар.остаток {r['invar_resid']:.3f} "
              f"| eigL1 {r['eigL1']} eigL2 {r['eigL2']} | lagλ {r['lag_eigs']} | ov u2→v1(L1,L2) {r['ov_u2_v1']}",flush=True)
(D/"EIGEN_e.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E76e DONE",flush=True)

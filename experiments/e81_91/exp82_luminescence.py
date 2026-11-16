#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-82 LUMINESCENCE: форма послеcвечения и тушение (аналогии физики люминесценции).
A) Форма затухания |ACF| own e1/e2 (лаги 1..24) — три модели по BIC:
   exp:        ln|A| = a − k/τ
   power(Рэндолл-Уилкинс): ln|A| = a − μ·ln k
   KWW:        ln|A| = a − (k/τ)^β, β∈(0.2,1] сеткой
   Честность: лаги, где |A|<0.02, из фитта вон; лагов всего ~12-16 — сила различения мала,
   вердикт «не разрешено» — легитимный исход.
B) Тушение (Stern-Volmer): тушащий Q = локальный сюрприз. Для каждого дока — события
   high-Q (p1 в нижнем квартилю = большой сюрприз) vs low-Q (верхний квартиль);
   «послесвечение после события» = среднее p(t0+k)·sign по событиям с фиксацией знака
   p(t0): S_cond(k)=⟨p(t0+k)·sign p(t0)⟩. Если high-Q быстрее роняет |S_cond| — тушение.
   Сравнение площади |S| по лагам 2..8, доковый бутстрап CI.
Null-гейт: та же статистика на случайных t0 (не сюрпризных) как фон.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e82"; OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(82); sfx="_L23"; KM=24
zo=np.load(D/"P1_own.npz"); zc=np.load(D/"P2_corpus.npz")
def rows(z):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
Uo=rows(zo); Uc=rows(zc)
C=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(C);B=V[:,-2:]
Xs=[];Ys=[]
for U in Uo+Uc:
    A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
M=np.linalg.solve(sum(x.T@x for x in Xs).T,sum(x.T@y for x,y in zip(Xs,Ys)).T).T
w,Vv=np.linalg.eigh(.5*(M+M.T));pl=Vv[:,np.argsort(w)[:2]]
def signser(U,kk):
    A=U@B;Dl=A[1:]-A[:-1];Dl/= (np.linalg.norm(Dl,axis=1,keepdims=True)+1e-12)
    return Dl@pl[:,kk]
P1=[signser(U,0) for U in Uo]; P2=[signser(U,1) for U in Uo]
def acf(p,K=KM):
    p=p-p.mean(); v=np.dot(p,p)+1e-12
    return np.array([np.dot(p[:-(k)],p[k:])/v for k in range(1,K+1)])
A={"own_e1":np.median([acf(p) for p in P1],0),"own_e2":np.median([acf(p) for p in P2],0)}
out={"acf":{k:[round(float(x),4) for x in v] for k,v in A.items()}}
def fit_models(v):
    m=np.abs(v)>=0.02; ks=np.arange(1,len(v)+1)[m]; ys=np.log(np.abs(v[m])); n=len(ks)
    if n<6: return None
    r={}
    for nm,X,y in (("exp",np.stack([ks,np.ones(n)]),ys),
                   ("power",np.stack([np.log(ks),np.ones(n)]),ys)):
        beta,res,_,_=np.linalg.lstsq(X.T.reshape(n,2),ys,rcond=None)
        pred=X.T@beta; rss=float(np.sum((ys-pred)**2))
        r[nm]=dict(bic=round(n*np.log(rss/n+1e-12)+2*np.log(n),2),tau_or_mu=round(float(beta[0]),3))
    best=None
    for be in (0.3,0.5,0.7,1.0):
        for lg_tau in np.linspace(0,3,40):
            tau=10**lg_tau
            pred=-(ks/tau)**be
            pred=pred- (np.mean(ys)-np.mean(pred))
            rss=float(np.sum((ys-pred)**2))
            bic=n*np.log(rss/n+1e-12)+3*np.log(n)
            if best is None or bic<best[0]: best=(bic,tau,be)
    r["kww"]=dict(bic=round(float(best[0]),2),tau=round(float(best[1]),2),beta=best[2])
    return r
for k,v in A.items():
    out[f"shape_{k}"]=fit_models(v)
    print(k,"shape:",json.dumps(out[f"shape_{k}"],ensure_ascii=False),flush=True)
# B) тушение: S_cond(k) после high-Q vs low-Q событий, фон=случайные t0
p1s=[-np.log(np.clip(zo[k].astype(np.float64),1e-6,1)) for k in sorted(f for f in zo.files if f.endswith("_p1"))]
def S_after(ps,surp,mode,K=8):
    ev=[]
    q1,q3=np.percentile(surp,[25,75])
    for p,s in zip(ps,p1s):
        T=min(len(p),len(s))-K
        for t0 in range(8,T):
            ok = (s[t0]>=q1 if mode=="highQ" else s[t0]<=q3 if mode=="lowQ" else True)
            if mode=="rand":
                if RNG.random()>0.25: continue
            if not ok: continue
            sg=np.sign(p[t0])
            if sg==0: continue
            ev.append(sg*p[t0+1:t0+K+1])
    E=np.array(ev); return float(np.mean(np.abs(E[:,1:8].mean(1)))), E.shape[0]
res={}
for cond in ("highQ","lowQ","rand"):
    a,n=S_after(P1,None,cond) if False else S_after(P1,p1s,cond)
    res[f"S_{cond}"]=round(a,4); print(f"S_{cond}={a:.4f} (n={n})",flush=True)
diff=[]
for _ in range(2000):
    hi,_=S_after(P1,p1s,"highQ"); lo,_=S_after(P1,p1s,"lowQ"); diff.append(hi-lo)
d=np.array(diff)
out["quench"]=dict(highQ=res["S_highQ"],lowQ=res["S_lowQ"],rand=res["S_rand"],
    hi_minus_lo=round(res["S_highQ"]-res["S_lowQ"],4),
    null=[round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)])
print("quench:",json.dumps(out["quench"]),flush=True)
(OUT/"LUMI.json").write_text(json.dumps(out,indent=1,ensure_ascii=False,default=float))
print("E82 DONE",flush=True)

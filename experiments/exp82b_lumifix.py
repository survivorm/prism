#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-82b LUMI-FIX: физически честная версия послеcвечения (интенсивность вместо знаковой ACF)
+ честный шумовой пол. Само-аудит e82 требовал: канон-плоскость R=16 (как e76k), шумовой
пол по энсемблю циклических сдвигов, фитт только по лагам где сигнал > 2×пол, и
INTENSITY-аналог: r(k) = E[-mode energy(t0+k)] / baseline после ИМПУЛЬСА (всплеск сюрприза),
против контроля случайных t0. Модели exp vs power — WLS по 1/σ_k² (σ — доковый бутстрап
по r(k)), ΔBIC по χ²; n_eff мал — «не разрешено» легитимно.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e82"; OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(8282); sfx="_L23"; KM=12; R=16
zo=np.load(D/"P1_own.npz"); zc=np.load(D/"P2_corpus.npz")
def rows(z):   # для базиса/плоскости — канон e76k (unit-строки, центр по столбцам)
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
Uo=rows(zo); Uc=rows(zc)
C=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(C);B=V[:,-R:]
Xs=[];Ys=[]
for U in Uo+Uc:
    A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
M=np.linalg.solve(sum(x.T@x for x in Xs).T,sum(x.T@y for x,y in zip(Xs,Ys)).T).T
w,Vv=np.linalg.eigh(.5*(M+M.T));pl=Vv[:,np.argsort(w)[:2]]
PB=B@pl                                # 1024×2 — модовые направления в сыром пространстве
# ---- сырые (не нормированные) траектории для интенсивности ----
def rawrows(z):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx): o.append(z[k].astype(np.float64))
    return o
Ro=rawrows(zo); assert len(Ro)==16
sfxp=[k for k in sorted(zo.files) if k.endswith("_p1")]
p1s=[-np.log(np.clip(zo[k].astype(np.float64),1e-6,1)) for k in sfxp]
# 1) шумовой пол знаковой ACF (канон R=16): сдвиги p
def signser(U):
    A=U@PB; Dl=A[1:]-A[:-1]; Dl/=(np.linalg.norm(Dl,axis=1,keepdims=True)+1e-12)
    return Dl
S=[signser(U) for U in Uo]
def acf_of_cols(P,K=KM):
    out=[]
    for kk in range(2):
        p=P[:,kk]-P[:,kk].mean(); v=np.dot(p,p)+1e-12
        out.append([np.dot(p[:-(k)],p[k:])/v for k in range(1,K+1)])
    return np.array(out)
obs=np.median([acf_of_cols(P) for P in S],0)          # (2,K)
floor=[]
for rep in range(400):
    fl=[]
    for P in RNG.permutation(S)[:6]:
        P2s=P*RNG.choice([-1,1],size=P.shape)   # sign-flip = честный null ACF (roll его сохраняет!)
        fl.append(acf_of_cols(P2s))
    floor.append(np.median(np.abs(np.array(fl)),0))
floor=np.mean(floor,0)*2.0                             # 2×полумедианный |пол|
out={"acf_obs":[[round(float(x),4) for x in r] for r in obs],
     "acf_floor2x":[[round(float(x),4) for x in r] for r in floor]}
def shape_fit(v,fl):
    m=np.abs(v)>fl; ks=np.arange(1,len(v)+1)[m]
    ys=np.log(np.abs(v[m])); n=int(m.sum())
    if n<5: return dict(verdict="не разрешено",n_eff=n)
    res={}
    for nm,xs,kpar in (("exp",ks,2),("power",np.log(ks),2)):
        X=np.stack([xs,np.ones(n)]).T
        beta,_,_,_=np.linalg.lstsq(X,ys,rcond=None)
        rss=float(np.sum((ys-X@beta)**2))
        res[nm]=dict(bic=round(n*np.log(rss/n+1e-12)+kpar*np.log(n),2),slope=round(float(beta[0]),3))
    a,b=res["exp"]["bic"],res["power"]["bic"]
    d=a-b
    res["verdict"]=("power (осторожно)" if d>6 else "exp" if d<-6 else "не разрешено")
    res["dBIC"]=round(d,2); res["n_eff"]=n; res["lags_used"]=[int(k) for k in ks]
    return res
for kk,nm in ((0,"e1"),(1,"e2")):
    out[f"shape_{nm}"]=shape_fit(obs[kk],floor[kk])
    print(nm,"shape:",json.dumps(out[f"shape_{nm}"],ensure_ascii=False),flush=True)
# 2) INTENSITY: r(k) после импульса (surprise ≥ p75) vs случайные t0
K=KM
def intensity(mode):   # mode: "pulse" | "rand"
    per=[]
    for H,p,s in zip(Ro,S,p1s):
        Dh=np.diff(H,axis=0)                     # сырые шаги
        e_all=np.sum(Dh*Dh,axis=1)+1e-12
        e_m=np.sum((Dh@PB)**2,axis=1)/e_all      # доля e-моды в шаге (безразмерно)
        Tm=min(len(e_m),len(s))-K
        q3=np.percentile(s[:Tm],75)
        if mode=="pulse": sel=np.where(s[:Tm]>=q3)[0]
        else:
            r0=RNG.random(Tm-8); sel=np.where(r0<0.15)[0]+8
        sel=sel[sel>=8]
        if len(sel)<20: per.append([np.nan]*K); continue
        base=np.median(e_m[8:Tm])
        per.append([float(np.median(e_m[sel+k])/ (base+1e-12)) for k in range(1,K+1)])
    return np.array(per)
rp=intensity("pulse"); rr=intensity("rand")
mp=np.nanmedian(rp,0); mr=np.nanmedian(rr,0)
# σ по лагам: доковый бутстрап медианы
sig=[]
for k in range(K):
    col=rp[:,k][~np.isnan(rp[:,k])]
    bs=[np.median(col[RNG.integers(0,len(col),len(col))]) for _ in range(800)]
    sig.append(float(np.std(bs)))
sig=np.array(sig)
out["intensity"]=dict(pulse=[round(float(x),4) for x in mp],rand=[round(float(x),4) for x in mr],
                      sigma=[round(float(x),4) for x in sig])
print("r_pulse:",np.round(mp,3).tolist()); print("r_rand :",np.round(mr,3).tolist())
print("sigma  :",np.round(sig,3).tolist(),flush=True)
# фитт спада (r-1) по лагам с SNR>2
y=mp-1.0; ks=np.arange(1,K+1)
m=(y>0)&(y/sig>2); ks=ks[m]; yy=y[m]; ss=sig[m]; n=len(ks)
if n>=4:
    res={}
    for nm,xs in (("exp",ks),("power",np.log(ks))):
        X=np.stack([xs,np.ones(n)]).T
        W=1.0/ss**2
        beta=np.linalg.solve(X.T@(X*W[:,None]),X.T@(yy*W))
        chi2=float(np.sum((yy-X@beta)**2*W))
        res[nm]=dict(chi2=round(chi2,2),bic=round(chi2+2*np.log(n),2),slope=round(float(beta[0]),3))
    d=res["exp"]["bic"]-res["power"]["bic"]
    res["verdict"]="power (осторожно)" if d>6 else ("exp" if d<-6 else "не разрешено")
    res["dBIC_exp_minus_power"]=round(d,2); res["n_eff"]=n
    out["intensity_shape"]=res
    print("intensity shape:",json.dumps(res,ensure_ascii=False),flush=True)
else:
    out["intensity_shape"]=dict(verdict="не разрешено",n_eff=n)
    print("intensity shape: не разрешено (n_eff=",n,")")
(OUT/"LUMI_FIX.json").write_text(json.dumps(out,indent=1,ensure_ascii=False,default=float))
print("E82B DONE",flush=True)

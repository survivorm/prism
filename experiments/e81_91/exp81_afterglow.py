#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-81 AFTERGLOW↔RECURRENCE: «послесвечение должно увеличивать вероятность повтора».
Только CPU, по P1_own/P2_corpus (+p1, +same1 из P1_own_meta). Состояния unit-норм.
Пре-рег (до запуска):
 R1: плотность возвращений dens = доля t с max_{τ∈[8,160]} cos(h_t,h_{t−τ})>θ
     выше в own, чем в corpus (оба θ; θ = p90/p95 пула own∪cor).
 R2: внутри own corr(e1(t), ev(t)) > 0 НЕ ждём; ждём corr(e1(t), ev(t+k)) ОТРИЦАТЕЛЬНУЮ
     на k=1..3 (антифазная энергия перед возвращением — «торможение у旧состояния»),
     ИЛИ положительную для e2. Что бы ни вышло — фиксируем обе моды и k=0..5.
 R3: события ev имеют БОЛЬШУЮ p1, чем не-события (повтор «уверенный») — own и cor.
 R4 (док-уровень, 16 точек): spearman(хвост e1 лаги2-5, dens_own) и spearman(хвост, same1)
     — если после-свечение питает повторы, знак +, CI через док-перестановки.
Null: θ-перестановки для R3 (перемешать ev по t внутри дока), для R4 — перестановка доков.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e81"; OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(81); L=23; sfx=f"_L{L}"; TAU=(8,160)
zo=np.load(D/"P1_own.npz"); zc=np.load(D/"P2_corpus.npz")
meta=json.load(open(D/"P1_own_meta.json"))
def rows(z):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
Uo=rows(zo); Uc=rows(zc); assert len(Uo)==16==len(Uc)
# общий базис+плоскость (канон e76k)
C=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(C);B=V[:,-2:]
Xs=[];Ys=[]
for U in Uo+Uc:
    A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
M=np.linalg.solve(sum(x.T@x for x in Xs).T,sum(x.T@y for x,y in zip(Xs,Ys)).T).T
w,Vv=np.linalg.eigh(.5*(M+M.T));pl=Vv[:,np.argsort(w)[:2]]
def e12(U):
    A=U@B;Dl=A[1:]-A[:-1];n2=np.sum(Dl*Dl,axis=1)+1e-12
    return np.stack([(Dl@pl[:,k])**2/n2 for k in range(2)],1)
def tail18(U):
    e=e12(U)[:,0]  # e1
    ee=e-e.mean(); v=np.dot(ee,ee)+1e-12
    ac=[np.dot(ee[:-k],ee[k:])/v for k in range(1,6)]
    return -np.mean(ac[1:4])   # «сила хвоста»: минус средних ACF(2..4)
def recmat(U):
    G=U@U.T; T=len(U)
    m=np.full(T,np.nan)
    for t in range(TAU[1],T):
        lo=max(0,t-TAU[1]); hi=t-TAU[0]
        if hi<=lo: continue
        m[t]=np.nanmax(G[t,lo:hi])
    return m
Mo=[recmat(U) for U in Uo]; Mc=[recmat(U) for U in Uc]
allv=np.concatenate([m[~np.isnan(m)] for m in Mo+Mc])
ths={"p90":float(np.percentile(allv,90)),"p95":float(np.percentile(allv,95))}
out={"theta":ths}
def dens(m,th): 
    v=m[~np.isnan(m)]; return float(np.mean(v>th))
for thn,th in ths.items():
    do=[dens(m,th) for m in Mo]; dc=[dens(m,th) for m in Mc]
    out[f"R1_{thn}"]=dict(own=round(float(np.median(do)),4),cor=round(float(np.median(dc)),4),
        frac_own_gt=round(float(np.mean(np.array(do)>np.array(dc))),3))
    print(f"R1 {thn}: own dens {np.median(do):.3f} vs cor {np.median(dc):.3f}; доля доков own>cor {np.mean(np.array(do)>np.array(dc)):.2f}",flush=True)
# R2: corr(e_k(t), ev(t+k))
for thn,th in ths.items():
    if thn!="p90": continue
    for cond,Us,Ms in (("own",Uo,Mo),("cor",Uc,Mc)):
        res={}
        for kk,ch in ((0,"e1"),(1,"e2")):
            row={}
            for k in range(6):
                xs=[];ys=[]
                for U,m in zip(Us,Ms):
                    e=e12(U)[:,kk]; T=min(len(e),len(m))
                    ev=(m[:T]>th).astype(float); ev[np.isnan(m[:T])]=np.nan
                    idx=np.arange(TAU[1],T-1-k)
                    xs.append(e[idx]); ys.append(ev[idx+k])
                x=np.concatenate(xs);y=np.concatenate(ys)
                ok=~np.isnan(y); x=x[ok];y=y[ok]
                row[f"k{k}"]=round(float(np.corrcoef(x,y)[0,1]),4)
            res[ch]=row
        out[f"R2_{cond}"]=res
        print(f"R2 {cond}: "+" ".join(f"{ch} "+" ".join(f"k{k}:{res[ch][f'k{k}']:+.3f}" for k in range(6)) for ch in ("e1","e2")),flush=True)
# R3: p1 на событиях vs нет (null: перестановка ev внутри дока)
zp=[-np.log(np.clip(zo[k].astype(np.float64),1e-6,1)) for k in sorted(f for f in zo.files if f.endswith("_p1"))]
for cond,Ms,S in (("own",Mo,zp),):
    diffs=[];nulls=[]
    for m,s in zip(Ms,S):
        T=min(len(m),len(s)); m=m[:T];s=s[:T]
        ok=~np.isnan(m)
        ev=m[ok]>ths["p90"]
        diffs.append(float(s[ok][ev].mean()-s[ok][~ev].mean()))
        nulls.append([float(s[ok][RNG.permutation(ev)].mean()-s[ok][~RNG.permutation(ev)].mean()) for _ in range(200)])
    obs=float(np.mean(diffs)); nl=np.concatenate(nulls)
    out["R3_own_p1_event_minus_nonevent"]=dict(obs=round(obs,4),
        null=[round(float(np.percentile(nl,2.5)),4),round(float(np.percentile(nl,97.5)),4)],
        signif=bool(obs<np.percentile(nl,2.5) or obs>np.percentile(nl,97.5)))
    print(f"R3 own: Δp1(событие−нет)={obs:+.4f} null[{np.percentile(nl,2.5):+.4f},{np.percentile(nl,97.5):+.4f}] {'SIGNIF' if out['R3_own_p1_event_minus_nonevent']['signif'] else '—'}",flush=True)
# R4: spearman по 16 own-докам: хвост vs dens, хвост vs same1
def spear(a,b):
    ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b))
    return float(np.corrcoef(ra,rb)[0,1])
tails=np.array([tail18(U) for U in Uo])
dens90=np.array([dens(m,ths["p90"]) for m in Mo])
same1=np.array([meta[f"own_w{i}"]["same1"] for i in range(16)])
out["R4_same1_order"]=[k.split("_w")[1].split("_")[0] for k in sorted(f for f in zo.files if f.endswith("_L23"))][:3]
for nm,y in (("dens",dens90),("same1",same1)):
    obs=spear(tails,y)
    nl=[spear(RNG.permutation(tails),y) for _ in range(2000)]
    out[f"R4_tail_vs_{nm}"]=dict(rho=round(obs,3),null=[round(float(np.percentile(nl,2.5)),3),round(float(np.percentile(nl,97.5)),3)])
    print(f"R4 tail vs {nm}: rho {obs:+.3f} null[{np.percentile(nl,2.5):+.3f},{np.percentile(nl,97.5):+.3f}]",flush=True)
(OUT/"AFTERGLOW_REC.json").write_text(json.dumps(out,indent=1,ensure_ascii=False,default=float))
print("E81 DONE",flush=True)

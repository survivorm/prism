#!/usr/bin/env python
"""B4-эффекты по температурам: что политика снимает. Для T∈{0.3,0.7,1.0}+T0:
(1) D-1 own (профиль k1..24, попарно-слойные corr внутри дока) против честного xdoc-контроля
    (та же L, разные невырожденные доки того же домена) — loopy-фильтр на ОБЕ стороны (урок P4);
(2) нить_Г по A4: cos-succ медиана единичных межслойных приращений vs crossdoc/bigshift (12 повторов);
(3) кросс-T перенос профиля: corr профиля тот-же-док/та-же-L между T (уникальная пара v5-пулов);
(4) глубина клетки на состояниях: cos(Δu(t),Δu(t')) по общим суффиксам k, в сравнении со случайными
    парами — на сэмплированных треках (токены из npz), L23.
САМОАУДИТ: u fp16→float64; кривые только k1..24; фильтр вырождения = cr(k1..96)max>0.95 по L23
(свой для каждого пула); контроль не видит вырожденные; пометить доки с суффиксным дефицитом."""
import json, os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np
D="/data/prism_kernels/exp62"; KM=24; KK=np.arange(1,97)
rng=np.random.default_rng(6404)
DOMS=["wiki","math","code","chat"]; LAY=[1,5,11,17,23]; PAIRS=[(1,5),(5,11),(11,17),(17,23)]
def dcurve(P,kk):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def Vpath(u):
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12); Vs=[]
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        Vs.append(vv/np.linalg.norm(vv) if np.linalg.norm(vv)>1e-6 else vv)
    return np.array(Vs)
def pc(a,b):
    a=a-a.mean(); b=b-b.mean(); return float(a@b/np.sqrt((a@a)*(b@b)+1e-30))
def load(Ttag,npz,metaf):
    z=np.load(npz); meta=json.load(open(metaf)); docs=meta.get("docs",meta)
    names=sorted(docs,key=lambda n:(DOMS.index(docs[n]["dom"]),docs[n]["si"]))
    u={n:{L:np.asarray(z[f"{n}_L{L}_u"],dtype=np.float64) for L in LAY} for n in names}
    tok={n:np.asarray(z[f"{n}_tok"]).tolist() for n in names}
    cr={n:dcurve(Vpath(u[n][23]),KK) for n in names}
    ok=[n for n in names if not cr[n][1:].max()>0.95]
    dom={n:docs[n]["dom"] for n in names}
    prof={n:{L:dcurve(Vpath(u[n][L]),np.arange(1,KM+1)) for L in LAY} for n in ok}
    return dict(tag=Ttag,names=names,ok=ok,dom=dom,u=u,tok=tok,prof=prof)
def d1ctrl(P):
    own=[];ctrl=[]
    for n in P["ok"]:
        for a,b in PAIRS: own.append(pc(P["prof"][n][a],P["prof"][n][b]))
    for rep in range(30):
        for n in P["ok"]:
            pool=[m for m in P["ok"] if P["dom"][m]==P["dom"][n] and m!=n]
            if not pool: continue
            m=pool[rng.integers(len(pool))]
            for a in [1,5,11,17,23]: ctrl.append(pc(P["prof"][n][a],P["prof"][m][a]))
    return round(float(np.median(own)),3),round(float(np.median(ctrl)),3),len(own),len(ctrl)
def threadA4(P,pair,reps=12):
    a,b=pair
    def g(n):
        A=P["u"][n][a]; B=P["u"][n][b]
        G=B-A; return G/np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-12)
    G={n:g(n) for n in P["ok"]}
    real={n:float(np.median(np.einsum("ij,ij->i",G[n][:-1],G[n][1:]))) for n in P["ok"]}
    nullcd=[];nullsh=[]
    for rep in range(reps):
        for n in P["ok"]:
            pool=[m for m in P["ok"] if P["dom"][m]==P["dom"][n] and m!=n]
            if pool:
                m=pool[rng.integers(len(pool))]; off=rng.integers(len(G[n]))
                idx=rng.integers(0,len(G[n]),300); j=(idx+off)%len(G[m])
                nullcd.append(float(np.median(np.einsum("ij,ij->i",G[n][idx],G[m][j]))))
            s=int(rng.integers(40,max(len(G[n])-40,41)))
            idx=rng.integers(0,max(len(G[n])-s,1),300)
            nullsh.append(float(np.median(np.einsum("ij,ij->i",G[n][idx],G[n][(idx+s)%len(G[n])]))))
    return {"real_med":round(float(np.median(list(real.values()))),3),
            "null_crossdoc":round(float(np.median(nullcd)),3),
            "null_bigshift":round(float(np.median(nullsh)),3)}
def celldepth(P):
    # суффиксы по сгенерированным токенам; позиции глобальные внутри дока (t от 4..T-2)
    from collections import defaultdict
    pos=defaultdict(list)
    for n in P["ok"]:
        t_=P["tok"][n]
        for i in range(4,len(t_)-2): pos[tuple(t_[i-2:i+1])].append((n,i))
    du={n:(lambda U:U[1:]-U[:-1])(P["u"][n][23]/np.maximum(np.linalg.norm(P["u"][n][23],axis=1,keepdims=True),1e-12)) for n in P["ok"]}
    out={}
    for k in [1,2,4,8]:
        cos_s=[]; agree=0; tot=0
        # суффикс длины k: токен-суффиксы [i-k..i] (k+1 токенов), шаг за шагом
        suf=defaultdict(list)
        for n in P["ok"]:
            t_=P["tok"][n]
            for i in range(k,len(t_)-2): suf[tuple(t_[i-k:i+1])].append((n,i))
        for key,ps in suf.items():
            if len(ps)<2: continue
            for x in range(min(len(ps),12)):
                for y in range(x+1,min(len(ps),12)):
                    (n1,i1),(n2,i2)=ps[x],ps[y]
                    if n1==n2 and abs(i1-i2)<3: continue
                    v1=du[n1][i1-1]; v2=du[n2][i2-1]
                    den=np.linalg.norm(v1)*np.linalg.norm(v2)
                    if den>1e-9: cos_s.append(float(v1@v2/den))
                    if P["tok"][n1][i1+1]==P["tok"][n2][i2+1]: agree+=1
                    tot+=1
        base=[]
        for _ in range(4000):
            n1,n2=rng.choice(P["ok"],2,replace=False)
            i1=rng.integers(4,len(du[n1])-1); i2=rng.integers(4,len(du[n2])-1)
            v1=du[n1][i1]; v2=du[n2][i2]; den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-9: base.append(float(v1@v2/den))
        out[f"k{k}"]={"pairs":tot,"cos_med":round(float(np.median(cos_s)),3) if cos_s else None,
                      "cos_base":round(float(np.median(base)),3),
                      "tok_agree":round(agree/max(tot,1),3)}
    return out
res=[]
P0=load("T0",f"{D}/gen_layers_qwen.npz",f"{D}/gen_layers_qwen_meta.json"); res.append(P0["tag"])
pools={"T0":P0}
for T in ["0.3","0.7","1.0"]:
    pools[f"T{T}"]=load(f"T{T}",f"{D}/gen_T{T}_qwen.npz",f"{D}/gen_T{T}_qwen_meta.json")
summary=[]
for tag in ["T0","T0.3","T0.7","T1.0"]:
    P=pools[tag]; o={"tag":tag,"n_ok":len(P["ok"])}
    o["d1_own"],o["d1_xdoc"],o["n_own"],o["n_ctrl"]=d1ctrl(P)
    for pair in [(1,5),(11,17),(17,23)]:
        o[f"thread_{pair[0]}_{pair[1]}"]=threadA4(P,pair)
        print(tag,pair,o[f"thread_{pair[0]}_{pair[1]}"],flush=True)
    o["cell"]=celldepth(P); print(tag,"cell",o["cell"],flush=True)
    summary.append(o)
# кросс-T перенос профиля (тот же док, та же L)
cross={}
for tag in ["T0.3","T0.7","T1.0"]:
    vals=[]
    for n in pools["T0"]["ok"]:
        if n not in pools[tag]["ok"]: continue
        for L in [5,11,23]: vals.append(pc(pools["T0"]["prof"][n][L],pools[tag]["prof"][n][L]))
    cross[f"T0_vs_{tag}"]={"n":len(vals),"med":round(float(np.median(vals)),3)}
summary.append({"tag":"crossT_profile_transfer","cross":cross})
json.dump(summary,open("../results/e64/B4_effects.json","w"),ensure_ascii=False,indent=1)
for o in summary: print(json.dumps(o,ensure_ascii=False)[:400],flush=True)
print("saved B4_effects.json")

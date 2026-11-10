#!/usr/bin/env python
"""AT-батарея (пре-рег §16): аттрактор режима после контекстов C1..C4.
Метрики на ПЕРЕСЕЧЕНИИ невырожденных доков (все условия + база): xdom / xdoc-same-dom / own
профилей k1..24 (L 5,11,23), dip/hump/plateau, loopy доля, lk4-hit.
Bootstrap по ПАРАМ документов, B=1000, для C1 − max(C2,C3,C4)."""
import json, os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np
D="/data/prism_kernels/exp62"; KM=24; KK=np.arange(1,97)
DOMS=["wiki","math","code","chat"]; LAY=[1,5,11,17,23]; PL=[5,11,23]
rng=np.random.default_rng(717)
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
def lk4(t):
    nxt={}; hit=np.zeros(len(t),dtype=bool)
    for i in range(3,len(t)-1):
        k=(int(t[i-3]),int(t[i-2]),int(t[i-1]),int(t[i]))
        pv=nxt.get(k)
        if pv is not None and pv==int(t[i+1]): hit[i]=True
        nxt[k]=int(t[i+1])
    return hit
def load(npz,metaf,names_want):
    z=np.load(npz); meta=json.load(open(metaf)); docs=meta.get("docs",meta)
    u={n:{L:np.asarray(z[f"{n}_L{L}_u"],dtype=np.float64) for L in PL} for n in names_want}
    tok={n:z[f"{n}_tok"].tolist() for n in names_want}
    cr={n:dcurve(Vpath(np.asarray(z[f"{n}_L23_u"],dtype=np.float64)),KK) for n in names_want}
    ok={n for n in names_want if not cr[n][1:].max()>0.95}
    prof={n:{L:dcurve(Vpath(u[n][L]),np.arange(1,KM+1)) for L in PL} for n in names_want}
    feats={n:{"dip":cr[n][0],"hump":cr[n][1:8].max(),"plat":float(np.median(cr[n][8:96])),
             "hit":float(lk4(tok[n])[4:].mean())} for n in names_want}
    return ok,prof,feats,{n:docs[n]["dom"] for n in names_want}
names=[f"{dm}_{si}" for dm in DOMS for si in range(8)]
base=load(f"{D}/gen_layers_qwen.npz",f"{D}/gen_layers_qwen_meta.json",names)
conds={"BASE":base}
for c in ["C1","C2","C3","C4"]:
    p=f"{D}/gen_AT_{c}_qwen.npz"
    if not os.path.exists(p): raise SystemExit(f"awaiting {p}")
    conds[c]=load(p,p.replace(".npz","_meta.json"),names)
inter=set.intersection(*[c[0] for c in conds.values()])
print("intersect nondeg:",sorted(inter),flush=True)
def xdoms(prof,dm,S):
    v=[]
    Sn=sorted(S)
    for i,a in enumerate(Sn):
        for b in Sn[i+1:]:
            if dm[a]==dm[b]: continue
            for L in PL: v.append(pc(prof[a][L],prof[b][L]))
    return np.array(v)
def xdoc(prof,dm,S):
    v=[]; Sn=sorted(S)
    for i,a in enumerate(Sn):
        for b in Sn[i+1:]:
            if dm[a]!=dm[b]: continue
            for L in PL: v.append(pc(prof[a][L],prof[b][L]))
    return np.array(v)
res={}
for c,(ok,prof,feats,dm) in conds.items():
    S=inter
    xd=xdoms(prof,dm,S); xs=xdoc(prof,dm,S)
    own=np.array([pc(prof[n][5],prof[n][23]) for n in S]+[pc(prof[n][11],prof[n][23]) for n in S])
    res[c]={"xdom_med":round(float(np.median(xd)),4),"xdom_ci":[round(float(x),4) for x in np.percentile(xd,[2.5,97.5])],
            "xdoc_med":round(float(np.median(xs)),4),"own_med":round(float(np.median(own)),4),
            "dip":round(float(np.median([feats[n]["dip"] for n in S])),4),
            "hump":round(float(np.median([feats[n]["hump"] for n in S])),4),
            "plat":round(float(np.median([feats[n]["plat"] for n in S])),4),
            "hit":round(float(np.median([feats[n]["hit"] for n in S])),3),
            "n_inter":len(S)}
    print(c,res[c],flush=True)
# bootstrap по документам: C1 − max(C2,C3,C4) на общих документах
S=sorted(inter)
def xd_pairwise(prof,dm):
    out={}
    for i,a in enumerate(S):
        for b in S[i+1:]:
            if dm[a]==dm[b]: continue
            out[(a,b)]=np.array([pc(prof[a][L],prof[b][L]) for L in PL])
    return out
XP={c:xd_pairwise(conds[c][1],conds[c][3]) for c in conds}
pairs=list(XP["BASE"].keys())
boot=[]
for _ in range(1000):
    idx=rng.integers(0,len(pairs),len(pairs)); ps=[pairs[i] for i in idx]
    m={c:float(np.median(np.concatenate([XP[c][p] for p in ps]))) for c in ["C1","C2","C3","C4"]}
    boot.append(m["C1"]-max(m["C2"],m["C3"],m["C4"]))
boot=np.array(boot)
res["verdict"]={"C1_minus_max_ctrl":round(float(np.median(boot)),4),
                "ci95":[round(float(x),4) for x in np.percentile(boot,[2.5,97.5])],
                "gate(>=0.02 & ci>0)":bool(np.median(boot)>=0.02 and np.percentile(boot,2.5)>0)}
# loopy/hit на ВСЕХ 32 документах по условиям (конкурентное объяснение)
res["pool_wide"]={c:{"loopy":round(1-len(ok)/32,3),"hit":round(float(np.median([conds[c][2][n]["hit"] for n in names])),3)}
                  for c,ok in [(c,conds[c][0]) for c in conds]}
json.dump(res,open("../results/e64/AT_battery.json","w"),indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))

#!/usr/bin/env python
"""S5 личное P1 «глубина ячейки»: программа E64_pre требует знать, что такое клетки карты
prefix→token. Прямой замер: для пар позиций с совпадающим суффиксом длины k —
(a) next-token agreement P(tok[t+1] одинаков | суффикс k);
(b) 1−cos состояний U_t1,U_t2; (c) cos тангенсов перехода V.
Кривые по k=0..32; baselines: k=0 = случайные пары. Бонус: k50/k90 по доменам —
«эффективная память карты». САМОАУДИТ: суффикс tok[t-k+1..t] НЕ включает tok[t+1] (иначе подтасовка);
пары per doc капаются (200 на ключ); V из той же curve_u с сохранением индексов; median per doc→dom→pool."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np, sys as _s; _s.path.insert(0,'.')
from exp62_batt_lib import curve_u
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"; KMAX=32
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
rng=np.random.default_rng(101)
def doc_metrics(nm):
    toks=np.asarray(z[f"{nm}_tok"]).ravel(); T=len(toks)
    U=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64)
    Vfull=curve_u(z[f"{nm}_L{L}_u"],z[f"{nm}_L{L}_hn"])
    # индекс V↔t: curve_u бросает пустые строки; восстановим mask
    u=U* np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)[:,None]
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    cos_next=np.zeros(T-1); valid=[]
    vt=0
    for t in range(T-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        if np.linalg.norm(vv)>1e-6: cos_next[t]=vv/np.linalg.norm(vv); valid.append(t)
    Vidx=np.array(valid)
    Vm={t:Vfull[i] for i,t in enumerate(Vidx)}
    out={}
    for k in range(0,KMAX+1):
        ag=[];dS=[];cT=[]
        groups={}
        if k==0:
            groups[()]=list(range(1,T-2))
        else:
            for t in range(k-1,T-1):
                groups.setdefault(tuple(toks[t-k+1:t+1].tolist()),[]).append(t)
        for key,ts in groups.items():
            if len(ts)<2: continue
            if len(ts)>30: ts=list(rng.choice(ts,30,replace=False))
            for ii in range(len(ts)):
                for jj in range(ii+1,len(ts)):
                    t1,t2=ts[ii],ts[jj]
                    if t1==t2: continue
                    ag.append(1.0 if toks[t1+1]==toks[t2+1] else 0.0)
                    dS.append(1.0-float(U[t1]@U[t2]/(np.linalg.norm(U[t1])*np.linalg.norm(U[t2])+1e-20)))
                    if t1 in Vm and t2 in Vm: cT.append(float(Vm[t1]@Vm[t2]))
        out[k]={"agree":float(np.mean(ag)) if ag else None,"dstate":float(np.median(dS)) if dS else None,
                "ctrans":float(np.median(cT)) if cT else None,"npairs":len(ag)}
    return out
per={}
for i,nm in enumerate(ok):
    per[nm]=doc_metrics(nm)
    if i%8==7: print("P1",i+1,flush=True)
def agg(ns,k,f):
    v=[per[n][k][f] for n in ns if per[n][k][f] is not None]
    return round(float(np.median(v)),3) if v else None
nw=[n for n in ok if dom[n]!="wiki"]
KS=[0,1,2,4,8,12,16,24,32]
summ={}
for dm in ["math","code","chat","pooled_nonwiki","ALL"]:
    ns=nw if dm=="pooled_nonwiki" else ([n for n in ok if dom[n]==dm] if dm!="ALL" else ok)
    summ[dm]={f:{str(k):agg(ns,k,f) for k in KS} for f in ["agree","dstate","ctrans"]}
    ag=[(k,agg(ns,k,"agree")) for k in KS]
    vals=[a for a in ag if a[1] is not None]; mx=max(v for _,v in vals)
    k50=next((k for k,v in vals if v>=0.5*mx),None); k90=next((k for k,v in vals if v>=0.9*mx),None)
    summ[dm]["max_agree"]=round(mx,3); summ[dm]["k50"]=k50; summ[dm]["k90"]=k90
out={"per_doc":{n:{str(k):per[n][k] for k in [0,4,16,32]} for n in ok},"summary":summ}
print(json.dumps(summ,ensure_ascii=False,indent=1)[:1800])
json.dump(out,open("../results/e64/P1_cell.json","w"),ensure_ascii=False); print("saved")

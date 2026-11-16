#!/usr/bin/env python
"""S5 личное P2: геометрия по РЕЖИМАМ политики (disk-only лайт-B2). Для каждого шага t класс:
hit = 4-gram lookup предсказал tok[t+1] (как в e64c/A3), иначе construct. Кривые cos(V_t,V_{t+k}):
hit-hit, construct-construct, mixed — только по парам в классе. Фичи dip/плато + bootstrap CI.
Если дип/плато живут в construct-подпоследовательности — двухмодальная картина S4v2 оживает на
состояниях; если одинаковы — кривая слепая к режиму (ещё один gạch в сторону «поля»).
САМОАУДИТ: hit[t] про переход t→t+1 (update после запроса); V с сохранением t; пары (t,t+k) требуют
оба валидных; класс mixed симметричен; median per doc→dom; wiki=1 не домен."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np, sys as _s; _s.path.insert(0,'.')
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"; KM=32
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
def lk4(toks):
    T=len(toks); nxt={}; hit=np.zeros(T,dtype=bool)
    for t in range(3,T-1):
        key=(int(toks[t-3]),int(toks[t-2]),int(toks[t-1]),int(toks[t]))
        pv=nxt.get(key)
        if pv is not None and pv==int(toks[t+1]): hit[t]=True
        nxt[key]=int(toks[t+1])
    return hit
def V_with_idx(nm):
    u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64); hn=np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    Vs={}; 
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]; nrm=np.linalg.norm(vv)
        if nrm>1e-6: Vs[t]=vv/nrm
    return Vs
rng=np.random.default_rng(202)
per={}
for i,nm in enumerate(ok):
    toks=np.asarray(z[f"{nm}_tok"]).ravel(); hit=lk4(toks); Vs=V_with_idx(nm); ts=sorted(Vs)
    cls={t:(1 if hit[t] else 0) for t in ts}
    curves={}
    for lab in ["hit","con","mix"]:
        c=[]
        for k in range(1,KM+1):
            vals=[]
            for t in ts:
                t2=t+k
                if t2 not in Vs: continue
                pair=(cls[t],cls[t2])
                if lab=="hit" and pair==(1,1): vals.append(Vs[t]@Vs[t2])
                elif lab=="con" and pair==(0,0): vals.append(Vs[t]@Vs[t2])
                elif lab=="mix" and pair[0]!=pair[1]: vals.append(Vs[t]@Vs[t2])
            c.append(float(np.median(vals)) if len(vals)>=10 else None)
        curves[lab]=c
    per[nm]={"curves":curves,"hit_frac":round(float(np.mean([cls[t] for t in ts])),3)}
    if i%8==7: print("P2",i+1,flush=True)
def boot_med(ns,lab,k,B=2000):
    v=[per[n]["curves"][lab][k-1] for n in ns if per[n]["curves"][lab][k-1] is not None]
    if len(v)<3: return None
    v=np.array(v); bs=np.median(v[rng.integers(0,len(v),(B,len(v)))],0)
    return [round(float(np.median(v)),3),round(float(np.quantile(bs,0.025)),3),round(float(np.quantile(bs,0.975)),3)]
nw=[n for n in ok if dom[n]!="wiki"]
out={"summary":{}}
for dm in ["math","code","chat","pooled_nonwiki"]:
    ns=nw if dm=="pooled_nonwiki" else [n for n in ok if dom[n]==dm]
    out["summary"][dm]={lab:{"dip":boot_med(ns,lab,1),"k4":boot_med(ns,lab,4),"plateau":boot_med(ns,lab,16),
                            "k32":boot_med(ns,lab,32)} for lab in ["hit","con","mix"]}
    out["summary"][dm]["hit_frac_med"]=round(float(np.median([per[n]["hit_frac"] for n in ns])),3)
out["per_doc"]={n:per[n] for n in ok}
for dm,row in out["summary"].items():
    print(dm,json.dumps(row,ensure_ascii=False)[:400])
json.dump(out,open("../results/e64/P2_regime.json","w"),ensure_ascii=False); print("saved")

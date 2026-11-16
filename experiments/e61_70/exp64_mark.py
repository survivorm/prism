#!/usr/bin/env python
"""E-64c: «две машины = одна» per-doc. Lookup coverage (n=4, gen-only) × per-doc ω̂
(слим-репрагон фита батареи на L23 U). Если Spearman(joint, osc/ω̂) устойчиво отрицателен —
«хеш-таблица + конструктор» подтверждён на точках; иначе — нет. Заодно закрывает
часть теста 8 (per-doc winners сохраняются впервые)."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
sys.path.insert(0,'.')
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"
z=np.load(NPZ, allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
K=np.arange(1,25)

# --- 1) per-doc lookup joint n=4 (gen-only контекст) ---
N=4
def joint(tok):
    hits=cand=0
    idx={}
    for i in range(len(tok)-N):
        idx.setdefault(tuple(tok[i:i+N]),[]).append(i)
    for t in range(2*N,len(tok)):
        key=tuple(tok[t-N:t]); occ=[i for i in idx.get(key,[]) if i+N<t]
        if not occ: continue
        cand+=1
        cont=[tok[i+N] for i in occ]
        pred=max(set(cont),key=cont.count)
        hits+= int(pred==tok[t])
    return (hits/max(cand,1), cand/max(len(tok)-2*N,1))
lk={n:joint(np.asarray(z[f"{n}_tok"],dtype=np.int64)) for n in names}

# --- 2) per-doc ω̂/winners (слим-ядро из exp62_batt_lib.run) ---
from exp62_batt_lib import curve_u  # noqa
WG=np.round(np.arange(0.10,2.001,0.02),3)
RG=np.round(np.arange(0.02,0.991,0.01),3); RGX=np.round(np.concatenate([RG,np.arange(0.992,0.99965,0.001)]),4)
RG2=RG[::2]; ONES=np.ones(len(K)); pairs2=[(a,b) for a in RG2 for b in RG2[RG2>a+0.02]]
rng=np.random.default_rng(11)
def wls(y,w,cols):
    G=np.stack(cols,1); co=np.linalg.lstsq(G*w[:,None]**.5,y*w**.5,rcond=None)[0]
    return float(np.sum(w*(y-G@co)**2))
fit={}
for i,nm in enumerate(names):
    u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64); hn=np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)
    V=curve_u(u,hn); y=[]; sg=[]
    for k in K:
        P=np.einsum("ij,ij->i",V[:-k],V[k:]); y.append(float(np.median(P)))
        # FIX audit2 D7: iid-ресемплинг позиций по АУТОКОРРЕЛИРОВАННОМУ ряду занижал σ
        # (=> w=1/σ² завышен, qosc/qexp систематически завышены) — moving-block, блок 16.
        BLK=16; nb=len(P)//BLK
        sel=np.concatenate([np.arange(b*BLK,(b+1)*BLK)[None] for b in rng.integers(0,nb,(200,8))]).ravel()
        sg.append(max(float(np.median(P[sel],axis=1).std()),2e-3))
    y=np.array(y); sd=np.array(sg); w=1.0/sd**2; tot=float(np.sum(w*y**2))
    best=None
    for om in WG:
        c,s=np.cos(om*K),np.sin(om*K)
        f=min(wls(y,w,[r_**K*c,r_**K*s,ONES]) for r_ in RGX)
        if best is None or f<best[0]: best=(f,float(om))
    be=min((wls(y,w,[a**K,b**K,ONES]),float(a),float(b)) for a,b in pairs2)
    fit[nm]={"om":best[1],"qosc":1-best[0]/tot,"qexp":1-be[0]/tot,
             # FIX audit2 D8: «win» = best-SSE из ~3300 фитов против ~300 пар без null-калибровки:
             # поле НЕ свидетельствует до порога «как часто osc выигрывает на шуме» (rnull-калибровка — в очереди)
             "win":"osc" if best[0]<be[0] else "exp","r2":be[1],"r2b":be[2]}
    if i%16==15: print("fit",i+1,flush=True)

# --- 3) корреляции (Spearman вручную) ---
def spearman(xs,ys):
    def rk(v):
        o=np.argsort(v); r=np.empty(len(v)); r[o]=np.arange(len(v)); return r
    a,b=rk(xs),rk(ys); a-=a.mean(); b-=b.mean()
    return float(a@b/np.sqrt((a@a)*(b@b)))
jointv=np.array([lk[n][0] for n in names]); omv=np.array([fit[n]["om"] for n in names])
osc=np.array([fit[n]["win"]=="osc" for n in names],dtype=float)
out={"spearman_joint_om":spearman(jointv,omv),"spearman_joint_osc":spearman(jointv,osc),
     "per_doc":{n:{"joint4":lk[n][0],"cand_frac":lk[n][1],"dom":dom[n],**fit[n]} for n in names},
     "by_dom":{d:{"joint_med":float(np.median([lk[n][0] for n in names if dom[n]==d])),
                  "om_med_osc":float(np.median([fit[n]["om"] for n in names if dom[n]==d and fit[n]["win"]=="osc"] or [float("nan")])),
                  "osc_frac":float(np.mean([fit[n]["win"]=="osc" for n in names if dom[n]==d]))} for d in ["wiki","math","code","chat"]}}
print("spearman joint~om:",round(out["spearman_joint_om"],3)," joint~osc:",round(out["spearman_joint_om_osc"] if False else out["spearman_joint_osc"],3))
json.dump(out,open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results/e64/e64c_markov.json"),"w"))  # FIX audit4 S5: от ROOT, не "../"
print("saved e64c_markov.json")

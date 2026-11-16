#!/usr/bin/env python
"""E-64 A5: расщепление Δ-кривой на нить_T (сглаженная компонента пути) и残 (остаток).
smooth = скользящее среднее по времени от ЕДИНИЧНЫХ строк u (окно 16, pad edge), resid = u_dir − smooth;
обе компоненты прогоняются через тот же curve_u/оценщик Δ(k). Перекрёст = full − smooth − resid.
Фит: r=0.98 фикс, cos/sin/const по k=1..24, сетка ω 0.10..2.00 шаг .02; osc_gain=1−sse_ω/sse_flat.
D-1 репликация: корреляция профилей (k=1..24) между L1/L11/L23 на компонентах (подвыборка 4/домен).
L23: все 64 дока; L1/L11: 16 доков (si 1..4/домен). САМОАУДИТ: curve_u сам нормирует строки — hn=ones;
полная кривая считается заново на unit-u (не t25 с hn), иначе перекрёст несопоставим."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np, sys as _s
_s.path.insert(0,'.')
from exp62_batt_lib import curve_u
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; KM=24; WIN=16
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
sub=[n for n in names if int(n.rsplit("_",1)[-1])<=4]
WG=np.round(np.arange(0.10,2.001,0.02),3); K=np.arange(1,KM+1)
def smooth_ma(X,w=WIN):
    c=np.cumsum(np.pad(X,((w//2,w//2),(0,0)),mode="edge"),axis=0)
    return (c[w:]-c[:-w])/w
def dcurve(P):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in range(1,KM+1)])
def oscfit(y):
    A0=np.ones((KM,1)); s0=float((y-y.mean())@(y-y.mean()))
    best=(None,np.inf)
    for om in WG:
        A=np.stack([np.cos(om*K),np.sin(om*K),np.ones(KM)],1)
        r=y-A@np.linalg.lstsq(A,y,rcond=None)[0]
        q=float(r@r)
        if q<best[1]: best=(om,q)
    return {"om":best[0],"gain":round(1-best[1]/max(s0,1e-12),3)}
out={"per_doc":{},"layers":{}}
for nm in names:
    u=np.asarray(z[f"{nm}_L23_u"],dtype=np.float64); u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    s=smooth_ma(u); rr=u-s
    comps={"full":curve_u(u,np.ones(len(u))),"smooth":curve_u(s,np.ones(len(s))),"resid":curve_u(rr,np.ones(len(rr)))}
    cur={c:dcurve(V) for c,V in comps.items()}
    out["per_doc"][nm]={"curves":{c:[round(float(x),4) for x in v] for c,v in cur.items()},
        "fit":{c:oscfit(cur[c]) for c in ["smooth","resid"]},
        "dom":dom[nm],"loopy":loopy(nm)}
    if names.index(nm)%16==15: print("A5",names.index(nm)+1,flush=True)
# профили по слоям для подвыборки
for L in ["1","11","23"]:
    out["layers"][L]={}
    for nm in sub:
        u=np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64); u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
        s=smooth_ma(u); rr=u-s
        out["layers"][L][nm]={"smooth":[round(float(x),4) for x in dcurve(curve_u(s,np.ones(len(s))))],
                              "resid":[round(float(x),4) for x in dcurve(curve_u(rr,np.ones(len(rr))))]}
        print("A5 lay",L,nm,flush=True)
# сводки
def med_c(c,k):
    v=[out["per_doc"][n]["curves"][c][k-1] for n in ok]
    return round(float(np.median(v)),4)
cross=[]; 
for k in range(1,KM+1):
    f=med_c("full",k); sm=med_c("smooth",k); rs=med_c("resid",k); cross.append(round(f-sm-rs,4))
sw={"median_curves":{c:[med_c(c,k) for k in range(1,KM+1)] for c in ["full","smooth","resid"]},"cross":cross,
 "gain_med":{c:round(float(np.median([out["per_doc"][n]["fit"][c]["gain"] for n in ok])),3) for c in ["smooth","resid"]},
 "om_med_osc":{c:(round(float(np.median([out["per_doc"][n]["fit"][c]["om"] for n in ok if out["per_doc"][n]["fit"][c]["gain"]>0.3])),2) if any(out["per_doc"][n]["fit"][c]["gain"]>0.3 for n in ok) else None) for c in ["smooth","resid"]}}
def pc(a,b):
    a=np.array(a)-np.mean(a); b=np.array(b)-np.mean(b)
    return float(a@b/np.sqrt((a@a)*(b@b)+1e-30))
d1={"smooth":[],"resid":[]}
for nm in sub:
    for c in ["smooth","resid"]:
        d1[c].append(pc(out["layers"]["1"][nm][c],out["layers"]["23"][nm][c]))
        d1[c].append(pc(out["layers"]["11"][nm][c],out["layers"]["23"][nm][c]))
sw["D1_profile_corr_Lpairs"]={c:(round(float(np.median(v)),2) if v else None) for c,v in d1.items()}
out["summary"]=sw
print(json.dumps(sw,ensure_ascii=False,indent=1))
json.dump(out,open("../results/e64/A5_thread.json","w"),ensure_ascii=False); print("saved")

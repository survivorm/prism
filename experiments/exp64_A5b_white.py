#!/usr/bin/env python
"""E-64 A5b: белый null resid-канала. curve_u на iid-строках маргинала resid (L23, все 64):
базлайн «белого» V-оценщика на высокочастотном пути — отделит истинный анти-полосу resid (−0.14) от геометрии."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np, sys as _s; _s.path.insert(0,'.')
from exp62_batt_lib import curve_u
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; KM=24; WIN=16
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
ok=[n for n in names if not (max(t25["curves_k1_96"]["real"][n][1:])>0.95 or n in("wiki_0","wiki_4"))]
def smooth_ma(X,w=WIN):
    c=np.cumsum(np.pad(X,((w//2,w//2),(0,0)),mode="edge"),axis=0)
    return (c[w:]-c[:-w])/w
def dcurve(P):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in range(1,KM+1)])
rng=np.random.default_rng(9)
real=[];white=[]
for nm in ok:
    u=np.asarray(z[f"{nm}_L23_u"],dtype=np.float64); u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    r=u-smooth_ma(u)
    real.append(dcurve(curve_u(r,np.ones(len(r)))))
    w=r[rng.integers(0,len(r),len(r))]
    white.append(dcurve(curve_u(w,np.ones(len(w)))))
real=np.array(real);white=np.array(white)
out={"real_med":[round(float(x),4) for x in np.median(real,0)],
     "white_med":[round(float(x),4) for x in np.median(white,0)],
     "excess":[round(float(x),4) for x in np.median(real-white,0)]}
print(json.dumps(out)); json.dump(out,open("../results/e64/A5b_white.json","w"))

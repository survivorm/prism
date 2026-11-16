#!/usr/bin/env python
"""S5 личное P4: «глобальный профиль» LFM3 против Qwen — количественно.
L23 кривые k=1..24; для каждой модели: own-doc corr, xdoc corr, отклонение от медианного
профиля (1−corr), сам медианный профиль. Если xdoc(LFM3)≈own — профиль один на всех: кривая
есть константа модели, «документных» профилей там нет."""
import json, os, sys
for v in ("OMP_NUM_THREADS",): os.environ.setdefault(v="2")
import numpy as np, sys as _s; _s.path.insert(0,'.')
from exp62_batt_lib import curve_u
def dcurve(P,kk):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def pc(a,b):
    a=a-np.mean(a); b=b-np.mean(b)
    return float(a@b/np.sqrt((a@a)*(b@b)+1e-30))
res={}
for tag,npz in [("qwen","/data/prism_kernels/exp62/gen_layers_qwen.npz"),
                ("lfm3","/data/prism_kernels/exp62/gen_layers_lfm3.npz")]:
    z=np.load(npz,allow_pickle=True); meta=json.load(open(npz.replace(".npz","_meta.json")))
    t25=json.load(open("../results/e63/t25_ekurve.json"))
    def loopyq(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
    ok=[n for n in meta if meta[n].get("sha_ok")]
    curves={}
    for n in ok:
        try:
            c=dcurve(curve_u(z[f"{n}_L23_u"],z[f"{n}_L23_hn"]),np.arange(1,25))
            if not loopyq(n) if tag=="qwen" else not bool(max(c[1:]>0.95)) and np.isfinite(c).all(): curves[n]=c
        except Exception: pass
    ns=list(curves); C=np.array([curves[n] for n in ns])
    med=np.median(C,0)
    xdoc=np.median([pc(curves[ns[i]],curves[ns[j]]) for i in range(len(ns)) for j in range(i+1,len(ns))])
    dev=[1-pc(curves[n],med) for n in ns]
    res[tag]={"n":len(ns),"xdoc_corr":round(float(xdoc),3),
              "dev_med":round(float(np.median(dev)),3),"dev_q90":round(float(np.quantile(dev,0.9)),3),
              "median_profile":[round(float(x),3) for x in med]}
print(json.dumps(res,ensure_ascii=False,indent=1))
json.dump(res,open("../results/e64/P4_global.json","w"),ensure_ascii=False); print("saved")

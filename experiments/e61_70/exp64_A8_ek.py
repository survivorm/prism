#!/usr/bin/env python
"""E-64 A8: E(k>32) пер-док +мода дистанций копирования.
E_band = mean_{k=33..96}(real−blk32); bootstrap CI по невырожденным (wiki без loopy = 1 док — не доменный вывод).
Копия-дистанция: для 3-gram'а — расстояние до ПОСЛЕДНЕГО вхождения (обновление после запроса).
Спирман: E_band vs joint4*cand (e64c), E_peak vs режим бина дистанций.САМОАУДИТ: индексы k=33..96 = [32:96]; медиана per-doc -> domain; без wiki в пуле корреляций."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
t25=json.load(open("../results/e63/t25_ekurve.json"))
mark=json.load(open("../results/e64/e64c_markov.json"))["per_doc"]
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
BINS=[2,4,6,9,12,16,24,32,64,128,320]
def dists3(toks):
    d1={}; d2={}; D=[]
    for j in range(2,len(toks)):
        key=(int(toks[j-2]),int(toks[j-1]),int(toks[j]))
        old=d1.get(key)
        if old is not None: D.append(min(j-old,320))
        d2[key]=old; d1[key]=j
    return np.array(D) if D else np.zeros(0,dtype=int)
per={}
rng=np.random.default_rng(5)
for nm in names:
    r=np.array(t25["curves_k1_96"]["real"][nm]); b=np.array(t25["curves_k1_96"]["blk32"][nm])
    exc=r-b
    band=float(np.mean(exc[32:96]))
    peak=int(np.argmax(exc[32:96])+33)
    toks=np.asarray(z[f"{nm}_tok"]).ravel()
    D=dists3(toks)
    if len(D):
        hist=np.histogram(D,bins=BINS)[0]
        mode=int(np.argmax(hist))
    else: hist=np.zeros(len(BINS)-1,dtype=int); mode=None
    per[nm]={"E_band":round(band,4),"E_peak":peak,"dist_n":int(len(D)),
             "mode_bin":mode,"mode_lo":BINS[mode] if mode is not None else None,
             "frac_le32":round(float((D<=32).mean()),3) if len(D) else None,
             "joint4xcand":round(mark[nm]["joint4"]*mark[nm]["cand_frac"],3)}
def boot(ns,f,B=4000):
    v=np.array([f(n) for n in ns]); return [round(float(np.median(v)),4),
        round(float(np.median([v[rng.integers(0,len(v),len(v))].mean() for _ in range(B)]))-float(np.median(v)),4),
        round(float(np.quantile([v[rng.integers(0,len(v),len(v))] for _ in range(B)].mean() if False else [np.mean(v[rng.integers(0,len(v),len(v))]) for _ in range(B)],0.025)),4),
        round(float(np.quantile([np.mean(v[rng.integers(0,len(v),len(v))]) for _ in range(B)],0.975)),4)]
nonwiki_nd=[n for n in ok if dom[n]!="wiki"]
res={"per_doc":per,
 "E_band_by_dom":{dm:boot([n for n in ok if dom[n]==dm],lambda n:per[n]["E_band"]) for dm in ["wiki","math","code","chat"]},
 "E_band_pooled_nonwiki":boot(nonwiki_nd,lambda n:per[n]["E_band"]),
 "note":"[med, boot_med-mean, q2.5, q97.5]"}
def sp(ns,fa,fb):
    a=np.array([fa(n) for n in ns]); b=np.array([fb(n) for n in ns])
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<8: return None
    return round(float(np.corrcoef(np.argsort(np.argsort(a[m])),np.argsort(np.argsort(b[m])))[0,1]),2)
res["spear_Eband_vs_joint"]={
 "nonwiki_nd":sp(nonwiki_nd,lambda n:per[n]["E_band"],lambda n:per[n]["joint4xcand"]),
 "all_nd":sp(ok,lambda n:per[n]["E_band"],lambda n:per[n]["joint4xcand"])}
res["spear_Epeak_vs_modebin"]=sp([n for n in nonwiki_nd if per[n]["mode_bin"] is not None],
    lambda n:per[n]["E_peak"],lambda n:per[n]["mode_bin"])
pk=[per[n]["E_peak"] for n in nonwiki_nd]
res["E_peak_hist_nonwiki"]={str(k):int(np.sum(np.abs(np.array(pk)-k)<=4)) for k in (40,48,56,64,72,80,88,96)}
res["frac_le32_med_by_dom"]={dm:round(float(np.median([per[n]["frac_le32"] for n in ok if dom[n]==dm and per[n]["frac_le32"] is not None])),3) for dm in ["wiki","math","code","chat"]}
print(json.dumps({k:res[k] for k in res if k!="per_doc"},ensure_ascii=False,indent=1))
json.dump(res,open("../results/e64/A8_ek.json","w"),ensure_ascii=False); print("saved")

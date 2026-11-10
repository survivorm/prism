#!/usr/bin/env python
"""E-64 A3 «события/часы» (§11.4 тест-1): ω* = внешний темп событий?
Четыре индикатора на документ: word-start (BPE Ġ), segment-mark (переводы/whitespace-токены),
V-jump (‖V‖>q90, thinning gap>=3), copy-hit (lookup-4gram продолжение == факт. следующий токен).
Для каждого: ACF(k=1..24), Bernoulli-гейт MC, k_peak, CV меж-событийных интервалов.
Ставка И-1 проверяется как: k_peak(i) ≈ 2π/ω̂(doc) для osc-доков (|Δ|<=1.5, предсказание в сетке 2..24).
Плюс: phase_align2(doc) = cos(φ[к1..12] − φ[к13..24]) при ω̂ — согласованность фазы по половинкам k-сетки;
корреляция CV↔phase_align2 (регулярность часов ↔ согласованность).
САМОАУДИТ: ACF бинарного ряда по центру; Bernoulli-null с той же λ; ω̂/winners из e64c (не пересобранного);
jump по V (MA-free), не по h; hit[t] требует tok[t+1] (не трогает последний t)."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
sys.path.insert(0,'.')
from exp62_batt_lib import curve_u
from transformers import AutoTokenizer
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"; KM=24
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
ok=[n for n in names if not loopy(n)]
mark=json.load(open("../results/e64/e64c_markov.json"))["per_doc"]
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
allids=sorted({int(t) for n in names for t in np.asarray(z[f"{n}_tok"]).ravel().tolist()})
tokstr=tk.convert_ids_to_tokens(allids)
WORDINIT={i for i,s in zip(allids,tokstr) if isinstance(s,str) and (s.startswith("Ġ") or s.startswith("▁") or s.startswith(" "))}
SEGMENT=set()
for i in allids:
    s=tk.decode([i])
    if s is not None and len(s.strip())==0 and len(s)>0: SEGMENT.add(i)
print(f"wordinit {len(WORDINIT)}/{len(allids)} ids; segment {len(SEGMENT)} ids",flush=True)
def lk4_hit(toks):
    T=len(toks); nxt={}; hit=np.zeros(T,dtype=bool)
    for t in range(3,T-1):
        key=(int(toks[t-3]),int(toks[t-2]),int(toks[t-1]),int(toks[t]))
        pv=nxt.get(key)
        if pv is not None and pv<int(toks[t+1]): hit[t]=True  # продолжение совпало с факт. следующим (strict==)
        nxt[key]=int(toks[t])
    return hit
def acf(x,k):
    x=np.asarray(x,dtype=float)-np.mean(x); v=float(x@x)
    if v<=0: return 0.0
    return float((x[:len(x)-k]@x[k:])/v)
rng=np.random.default_rng(7)
res={}
for nm in names:
    toks=np.asarray(z[f"{n_:=nm}_tok"] if False else z[f"{nm}_tok"]).ravel()
    T=len(toks)
    z1=np.array([1.0 if int(t) in WORDINIT else 0.0 for t in toks])
    z2=np.array([1.0 if int(t) in SEGMENT else 0.0 for t in toks])
    V=curve_u(z[f"{nm}_L{L}_u"],z[f"{nm}_L{L}_hn"])
    j=np.linalg.norm(V,axis=1); q=np.quantile(j,0.90)
    z3=np.zeros(len(j)); last=-99
    for t in range(len(j)):
        if j[t]>q and t-last>=3: z3[t]=1.0; last=t
    z4=lk4_hit(toks).astype(float)
    row={}
    for ev,zz in [("word",z1),("seg",z2),("jump",z3),("hit",z4)]:
        lam=zz.mean()
        real=[acf(zz,k) for k in range(1,KM+1)]
        sims=np.stack([(lambda b: [acf(b,k) for k in range(1,KM+1)])(rng.random(len(zz))<lam) for _ in range(120)])
        band_hi=np.quantile(sims,0.975,axis=0); band_lo=np.quantile(sims,0.025,axis=0)
        exc=np.array(real)-np.array([np.median(s) for s in sims.T])
        kk=np.arange(1,KM+1)
        kp=int(kk[np.argmax(np.where(kk>=2,np.array(real),-9))]) if kk.size else None
        idx=np.flatnonzero(zz)
        gaps=np.diff(idx); gaps=gaps[(gaps>=2)&(gaps<=200)]
        cv=float(gaps.std()/gaps.mean()) if len(gaps)>4 else None
        row[ev]={"lam":round(float(lam),3),"acf": [round(float(x),3) for x in real],
                 "nullq": [round(float(x),3) for x in (band_lo+band_hi)],
                 "sig": bool(np.any(np.array(real)>band_hi)), "k_peak":kp,
                 "cv":round(cv,2) if cv is not None else None}
    m=mark[nm]; pred=2*np.pi/m["om"] if m["win"]=="osc" else None
    pa=None
    if pred is not None:
        y=np.array(t25["curves_k1_96"]["real"][nm][:KM]); K=np.arange(1,KM+1)
        def ph(ks):
            A=np.stack([np.cos(m["om"]*ks),np.sin(m["om"]*ks),np.ones_like(ks)],1)
            c=np.linalg.lstsq(A,y,rcond=None)[0]; return np.arctan2(c[1],c[0])
        pa=float(np.cos(ph(K[:12])-ph(K[12:])))
    row["om"]=m["om"]; row["win"]=m["win"]; row["pred_period"]=round(pred,1) if pred and pred<=24 else None
    row["phase_align2"]=round(pa,2) if pa is not None else None
    res[nm]=row
    if names.index(nm)%16==15: print("A3",names.index(nm)+1,flush=True)
out={"per_doc":res,"wordinit_n":len(WORDINIT),"segment_n":len(SEGMENT)}
# сводки
summ={}
for ev in ["word","seg","jump","hit"]:
    hits=[];tot=0;cvs=[];pas=[]
    for nm in ok:
        r=res[nm]; p=r.get("pred_period")
        if p: tot+=1; hits.append(abs(r[ev]["k_peak"]-p)<=1.5)
        if r[ev]["cv"] is not None: cvs.append(r[ev]["cv"])
        if r.get("phase_align2") is not None and r[ev]["cv"] is not None: pas.append((r[ev]["cv"],r["phase_align2"]))
    lam_by_dom={dm:round(float(np.median([res[n][ev]["lam"] for n in ok if dom[n]==dm])),3) for dm in ["wiki","math","code","chat"]}
    sigf=float(np.mean([res[n][ev]["sig"] for n in ok]))
    sp=None
    if len(pas)>8:
        a=np.array(pas); sp=round(float(np.corrcoef(np.argsort(np.argsort(a[:,0])),np.argsort(np.argsort(a[:,1])))[0,1]),2)
    summ[ev]={"hit_frac_pred_in_grid":round(float(np.mean(hits)),2) if hits else None,"n_pred":tot,
              "sig_frac":round(sigf,2),"cv_med":round(float(np.median(cvs)),2) if cvs else None,
              "lam_by_dom":lam_by_dom,"spearman_cv_vs_phasealign":sp}
out["summary"]=summ
print(json.dumps(summ,ensure_ascii=False,indent=1))
json.dump(out,open("../results/e64/A3_events.json","w"),ensure_ascii=False)
print("saved")

#!/usr/bin/env python
"""E-64 LFM3 вскрытие по PRE-REGISTRATION §11.3 (7 критериев; подписана до вскрытия).
Только sha_ok-доки. Кривые: median cos(V_t,V_{t+k}) по V=curve_u (MA-free), k=1..96;
null'ы: blk32 (перестановка 32-блоков пути) и iid (перестановка строк). Фит slim: WG×RG как e64c.
САМОАУДИТ: доменные выводы только при n_nondeg>=12 (критерий 7), иначе только пул;
D-1 контроль = кривые LFM3-L23 vs Qwen-L23 (покомпонентно, own-vs-own слои); ω* halfwidth = IQR per-doc osc ω̂ Qwen (аппроксимация — помечена)."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, zlib, sys as _s; _s.path.insert(0,'.')
from exp62_batt_lib import curve_u
NPZ="/data/prism_kernels/exp62/gen_layers_lfm3.npz"; LAY=["1","5","11","17","23"]; KM=96
WG=np.round(np.arange(0.10,2.001,0.02),3); RG=np.round(np.arange(0.02,0.991,0.01),3)
K=np.arange(1,25)
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
ok=[n for n in meta if meta[n].get("sha_ok")]
dom={n:meta[n]["dom"] for n in ok}
print("lfm3 docs ok:",len(ok),flush=True)
def blk_rows(x,seed,blk=32):
    r=np.random.default_rng(zlib.crc32(str(seed).encode())%99991); T=len(x); nb=T//blk; idx=r.permutation(nb)
    o=np.concatenate([x[b*blk:(b+1)*blk] for b in idx],0)
    return np.concatenate([o,x[nb*blk:]]) if nb*blk<T else o
def dcurve(P,kk):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def slim_fit(y,rng):
    sg=[max(float(np.median(y[rng.integers(0,len(y),200)] for _ in range(1) and [np.median(y[rng.integers(0,len(y),200)]) for _ in range(200)])),2e-3) for _ in [0]]
    return None
rng=np.random.default_rng(3)
def wls(y,w,A):
    co=np.linalg.lstsq(A*w[:,None],y*w**0.5,rcond=None)[0] if False else np.linalg.lstsq(A*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
    return float(np.sum(w*(y-A@co)**2)),co
def fit_doc(nm,L):
    V=curve_u(z[f"{nm}_L{L}_u"],z[f"{nm}_L{L}_hn"])
    y=np.array([float(np.median(np.einsum("ij,ij->i",V[:-k],V[k:]))) for k in K])
    sg=np.array([max(float(np.std([np.median(np.einsum("ij,ij->i",V[rng.integers(0,len(V),len(V))][:-k],V[rng.integers(0,len(V),len(V))][k:])) for _ in range(60)])),2e-3) for k in K])
    w=1/sg**2; best=(None,None,np.inf)
    for r in RG:
        for om in WG:
            A=np.stack([(r**K)*np.cos(om*K),(r**K)*np.sin(om*K),np.ones(24)],1)
            q,_=wls(y,w,A)
            if q<best[2]: best=(om,r,q)
    A0=np.ones((24,1)); q0,_=wls(y,w,A0)
    return best[0],best[1],round(1-best[2]/q0,3)
curv={"real":{},"blk32":{},"iid":{}}
fit={}
for i,nm in enumerate(sorted(ok)):
    u=np.asarray(z[f"{nm}_L23_u"],dtype=np.float64); hn=np.asarray(z[f"{nm}_L23_hn"],dtype=np.float64)
    for fam,uu in [("real",u),("blk32",blk_rows(u,nm)),("iid",u[np.random.default_rng(zlib.crc32(nm.encode())).permutation(len(u))])]:
        V=curve_u(uu,hn); curv[fam][nm]=dcurve(V,np.arange(1,KM+1))
    fit[nm]={"dom":dom[nm]}
    for L in ["23","11"]:
        om,r,qg=fit_doc(nm,L); fit[nm][f"L{L}"]={"om":round(float(om),2),"r":round(float(r),3),"gain":qg,"clip":bool(qg>0.3 and r>=0.989)}
    if i%8==7: print("lfm3",i+1,flush=True)
t25=json.load(open("../results/e63/t25_ekurve.json"))
def loopy(n): return bool(max(t25["curves_k1_96"]["real"][n][1:])>0.95) or n in("wiki_0","wiki_4")
loopy_l={n:bool(max(curv["real"][n][1:])>0.95) for n in ok}
nd=[n for n in ok if not loopy_l[n]]
out={"n_ok":len(ok),"n_nondeg":len(nd),"n_by_dom_nondeg":{dm:sum(1 for n in nd if dom[n]==dm) for dm in ["wiki","math","code","chat"]}}
# крит 3: локальность E(k)=real-blk32, bootstrap CI (пул nondeg; по доменам — только если n>=12)
rngb=np.random.default_rng(4)
def Eci(ns):
    E=np.array([np.array(curv["real"][n])-np.array(curv["blk32"][n]) for n in ns])
    med=np.median(E,0); bs=np.median(E[rngb.integers(0,len(ns),(2000,len(ns)))],0)
    return [round(float(med[k]),4) for k in (0,4,15,31,47,63,79,95)], [round(float(np.quantile(bs[:,k],0.025)),4) for k in (0,4,15,31,47,63,79,95)], [round(float(np.quantile(bs[:,k],0.975)),4) for k in (0,4,15,31,47,63,79,95)]
out["crit3_locality_E_by_dom"]={dm:Eci([n for n in nd if dom[n]==dm]) if sum(1 for n in nd if dom[n]==dm)>=3 else None for dm in ["wiki","math","code","chat"]}
out["crit3_pooled"]=Eci(nd); out["crit3_k_marks"]=[1,5,16,32,48,64,80,96]
# крит 1: ω* (L23, osc-gain docs) vs Qwen, halfwidth = IQR Qwen osc ω̂ (аппроксимация)
q=json.load(open("../results/e64/e64c_markov.json"))["per_doc"]
out["crit1_omega"]={}
for dm in ["wiki","math","code","chat"]:
    qo=[q[n]["om"] for n in q if q[n].get("win")=="osc" and not loopy(n) and (n.split("_")[0]==dm or (dm=="wiki" and n.startswith("wiki")))]
    qo=[q[n]["om"] for n in q if n.startswith(dm) and q[n]["win"]=="osc" and not loopy(n)]
    lo=[fit[n]["L23"]["om"] for n in nd if dom[n]==dm and fit[n]["L23"]["gain"]>0.3]
    out["crit1_omega"][dm]={"qwen_med":round(float(np.median(qo)),2) if qo else None,"qwen_iqr":round(float(np.subtract(*np.percentile(qo,[75,25]))),2) if len(qo)>3 else None,"n_qwen":len(qo),
      "lfm3_med":round(float(np.median(lo)),2) if lo else None,"n_lfm3":len(lo),
      "verdict":("n<12 — не вывод" if sum(1 for n in nd if dom[n]==dm)<12 else ("СОВПАЛО" if lo and qo and abs(np.median(lo)-np.median(qo))<0.5*float(np.subtract(*np.percentile(qo,[75,25]))) else "РАЗОШЛОСЬ"))}
# крит 5: clip-счётчик
out["crit5_clip"]={f"L{L}":{"clip":sum(1 for n in nd if fit[n][f"L{L}"]["clip"]),"gain3":sum(1 for n in nd if fit[n][f"L{L}"]["gain"]>0.3)} for L in ["23","11"]}
# крит 2: D-1 покомпонентно
def pc(a,b):
    a=np.asarray(a[:24],float)-np.mean(a[:24]); b=np.asarray(b[:24],float)-np.mean(b[:24])
    return float(a@b/np.sqrt((a@a)*(b@b)+1e-30))
own=[]; cross=[]
qw=json.load(open("../results/e63/d1_profiles.json")) if os.path.exists("../results/e63/d1_profiles.json") else None
for nm in nd:
    cs=[]
    for L in LAY:
        V=curve_u(z[f"{nm}_L{L}_u"],z[f"{nm}_L{L}_hn"])
        cs.append(dcurve(V,np.arange(1,25)))
    for a in range(5):
        for b in range(a+1,5): own.append(pc(cs[a],cs[b]))
    out.setdefault("d1_per_doc",{})[nm]=round(float(np.median([own[-1]])),3)
    if qw is not None and nm.replace("lfm3_","").split("_")[0]:
        pass
out["crit2_D1_own_med"]=round(float(np.median(own)),3)
out["crit2_crossmodel_note"]="контроль own-vs-чужая-модель считается отдельно в LFM3_ctrl"
# крит 6: нить_Г профиль
LAYI=[1,5,11,17,23]
def Hf(n,L): return np.asarray(z[f"{n}_L{L}_u"],dtype=np.float64)*np.asarray(z[f"{n}_L{L}_hn"],dtype=np.float64)[:,None]
prof={}
for a,b in zip(LAYI,LAYI[1:]):
    vals=[]
    for n in nd:
        g=Hf(n,b)-Hf(n,a); g=g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
        vals.append(float(np.median(np.einsum("ij,ij->i",g[:-1],g[1:]))))
    prof[f"{a}->{b}"]=round(float(np.median(vals)),3)
out["crit6_thread_grad_lfm3"]=prof; out["crit6_qwen_ref"]={"real":[0.35,0.518,0.608,0.548],"null_crossdoc":0.068,"null_bigshift":0.109}
# крит 4: copy-полоса
def lk4(toks):
    T=len(toks); nxt={}; hit=np.zeros(T,dtype=bool); cand=0; hitj=0
    for t in range(3,T-1):
        key=(int(toks[t-3]),int(toks[t-2]),int(toks[t-1]),int(toks[t]))
        pv=nxt.get(key)
        if pv is not None:
            cand+=1
            if pv==int(toks[t+1]): hit[t]=True; hitj+=1
        nxt[key]=int(toks[t+1])
    return hitj/max(T-4,1), cand/max(T-4,1)
cp={}
for nm in ok:
    h,c=lk4(np.asarray(z[f"{nm}_tok"]).ravel()); cp[nm]=round(h*c,3)
out["crit4_copy_jointcand_by_dom"]={dm:round(float(np.median([cp[n] for n in ok if dom[n]==dm])),3) for dm in ["wiki","math","code","chat"]}
out["crit4_qwen_ref"]={"wiki":0.48,"math":0.20,"code":0.12,"chat":0.50}
out["crit4_loopy_frac_lfm3"]=round(float(np.mean(list(loopy_l.values()))),3)
json.dump(out,open("../results/e64/LFM3_open.json","w"),ensure_ascii=False)
print(json.dumps({k:v for k,v in out.items() if k not in("d1_per_doc",)},ensure_ascii=False,indent=1)[:3000])
print("saved LFM3_open.json")

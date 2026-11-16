#!/usr/bin/env python
"""E-64 A7: feasibility + маргинальный канал (top1−top2, p1) по доменам (pre-регистрация S3).
logits_t = rmsnorm(h_t ⊙ ln_w) @ E^T (lm_head tied с embed_tokens).
FEASIBILITY-GATE: долю строк с argmax==факт. следующий токен считаем ДО маржин; <0.6 = прокси
не годится для «уверенности» (e64a: argmax != сгенер. токен в 37% случаев — там сравнивали НЕ то).
Выравнивание выбирается эмпирически среди {h_t→tok[t+1], h_t→tok[t]} × {с norm, без norm} (4 гипотезы).
САМОАУДИТ: h = u·hn (иерархия из npz); rms по строке; chunked matmul; softmax по полным логонам."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"6")
import numpy as np
from safetensors.numpy import load_file
D="/data/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
E=load_file(D)["model.language_model.embed_tokens.weight"].astype(np.float32)
LN=load_file(D)["model.language_model.norm.weight"].astype(np.float32)
V,Eh=E.shape; print("E",E.shape,flush=True)
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
def rmsn(X,w):
    return (X/np.sqrt((X*X).mean(1,keepdims=True)+1e-6))*w
def logits_chunk(Ht):
    Z=np.empty((len(Ht),V),dtype=np.float32)
    for a in range(0,V,512):
        Z[:,a:a+512]=Ht@E[a:a+512].T
    return Z
# --- feasibility: гипотезы на 4 докуках
docs4=names[::16][:4]
best=None
for use_ln in (True,False):
    for off in (1,0):
        m=[]
        for nm in docs4:
            h=(np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float32)*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float32)[:,None])
            X=rmsn(h,LN) if use_ln else h
            top=np.empty(len(X),dtype=int)
            for a in range(0,V,1024):
                Z=X@E[a:a+1024].T
                if a==0: top1=Z.max(1); top=np.argmax(Z,1)
                else:
                    mx=Z.max(1); upd=mx>top1; top[upd]=a+np.argmax(Z[upd],1); top1=np.maximum(top1,mx)
            tok=np.asarray(z[f"{nm}_tok"]).ravel()
            T=min(len(X)-1,len(tok)-1)
            if off==1: m.append(float((top[:T]==tok[1:T+1]).mean()))
            else: m.append(float((top[1:T+1]==tok[1:T+1]).mean()))
        r=float(np.mean(m)); print(f"hyp ln={use_ln} off={off}: match={r:.3f}",flush=True)
        if best is None or r>best[0]: best=(r,use_ln,off)
print("BEST",best,flush=True)
rate,use_ln,off=best
if rate<0.60:
    json.dump({"feasibility":rate,"verdict":"FAIL — прокси не годится"},open("../results/e64/A7_margin.json","w"))
    print("FAIL gate"); sys.exit(0)
# --- маржин: top1−top2 и p1 по всем 64
out={}
for i,nm in enumerate(names):
    h=(np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float32)*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float32)[:,None])
    X=rmsn(h,LN) if use_ln else h
    T=len(X)
    mx1=np.full(T,-1e30,dtype=np.float32); mx2=np.full(T,-1e30,dtype=np.float32)
    se=np.zeros(T,dtype=np.float64)
    sse=np.zeros(T,dtype=np.float64)
    for a in range(0,V,512):
        Z=(X@E[a:a+512].T).astype(np.float32)
        m=Z.max(1)
        mx2=np.maximum(mx2,np.minimum(mx1,m)); mx1=np.maximum(mx1,m)
        sse+=np.exp(Z.astype(np.float64)-mx1[:,None]).sum(1)*0  # placeholder, two-pass below
        Z=None
    # two-pass softmax sum с финальным mx1
    for a in range(0,V,4096):
        Z=(X@E[a:a+4096].T).astype(np.float64)
        se+=np.exp(Z-mx1[:,None]).sum(1)
        del Z
    p1=np.exp(mx1.astype(np.float64))/se
    out[nm]={"dom":dom[nm],"margin_med":round(float(np.median(mx1-mx2)),3),
             "p1_med":round(float(np.median(p1)),4),"p1_mean":round(float(p1.mean()),4),"feas_rate_used":round(rate,3)}
    if i%8==7: print("A7",i+1,round(float(np.median(mx1-mx2)),2),flush=True)
by={}
for dm in ["wiki","math","code","chat"]:
    ns=[n for n in names if dom[n]==dm]
    by[dm]={"margin_med":round(float(np.median([out[n]["margin_med"] for n in ns])),3),
            "p1_med":round(float(np.median([out[n]["p1_med"] for n in ns])),4)}
out["BY_DOMAIN"]=by
print(json.dumps(by,ensure_ascii=False))
json.dump(out,open("../results/e64/A7_margin.json","w"),ensure_ascii=False); print("saved")

#!/usr/bin/env python
"""E-64 A7: маргинальный канал (top1−top2, p1) по доменам (pre-регистрация S3).
Формула хэда ВЕРИФИЦИРОВАНА против живых model().logits (случайный вход):
  logits = rms(h_L23) @ E^T БЕЗ norm.weight — cos 1.000004; варианты w и 1+w — мимо (Qwen3_5RMSNorm).
Выравнивание ВЕРИФИЦИРОВАНО на пуле: argmax(row_t)==tok[t] (строки — состояние ДО tok[t]);
смещённая гипотеза tok[t+1] давала 0.234 — старые «расхождения argmax-прокси» были off-by-one.
Гейт повторяется на 2 докуках перед маржин (быстро). rms-скейлинг обязателен: без него маржин
масштабируется rms(h_t) — меженный сдвиг по ‖h‖ между доменами.
САМОАУДИТ: h = u·hn; chunked matmul; two-pass softmax по полным логонам; Z только fp32."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"6")
import numpy as np
import torch
from safetensors.torch import load_file as tload
D="/data/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
_sd=tload(D)
E=_sd["model.language_model.embed_tokens.weight"].float().numpy(); del _sd
V,Eh=E.shape; CH=512; nc=(V+CH-1)//CH
print("E",E.shape,flush=True)
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"
z=np.load(NPZ,allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))
def headX(h):  # точная копия верифицированного хэда
    return h/np.sqrt((h*h).mean(1,keepdims=True)+1e-6)
# --- гейт повтора (2 дока, off=0)
gate=[]
for nm in names[::32][:2]:
    h=(np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float32)*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float32)[:,None])
    X=headX(h)
    top=np.empty(len(X),dtype=int); top1=np.full(len(X),-1e30,dtype=np.float32)
    for a in range(0,V,1024):
        Z=X@E[a:a+1024].T
        mx=Z.max(1); upd=mx>top1; top[upd]=a+np.argmax(Z[upd],1); top1=np.maximum(top1,mx)
    tok=np.asarray(z[f"{nm}_tok"]).ravel(); T=min(len(X),len(tok))
    gate.append(float((top[1:T]==tok[1:T]).mean()))
rate=float(np.mean(gate)); print("gate match:",round(rate,3),flush=True)
if rate<0.60:
    json.dump({"feasibility":rate,"verdict":"FAIL"},open("../results/e64/A7_margin.json","w")); sys.exit(0)
out={}
for i,nm in enumerate(names):
    h=(np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float32)*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float32)[:,None])
    X=headX(h); T=len(X)
    Z=np.empty((T,V),dtype=np.float32)
    for a in range(0,V,CH):
        Z[:,a:a+CH]=X@E[a:a+CH].T
    cm=Z.reshape(T,nc,CH).max(2)
    part=np.partition(cm,-2,axis=1); mx1=part[:,-1]; mx2=part[:,-2]
    se=np.zeros(T,dtype=np.float64)
    for a0 in range(0,V,4096):
        se+=np.exp(Z[:,a0:a0+4096].astype(np.float64)-mx1[:,None]).sum(1)
    p1=1.0/se  # softmax top1 = e^{mx1}/Σe^z = 1/Σe^{z−mx1}
    del Z
    out[nm]={"dom":dom[nm],"margin_med":round(float(np.median(mx1-mx2)),3),
             "p1_med":round(float(np.median(p1)),4),"p1_mean":round(float(p1.mean()),4)}
    if i%8==7: print("A7",i+1,round(float(np.median(mx1-mx2)),2),flush=True)
by={}
for dm in ["wiki","math","code","chat"]:
    ns=[n for n in names if dom[n]==dm]
    by[dm]={"margin_med":round(float(np.median([out[n]["margin_med"] for n in ns])),3),
            "p1_med":round(float(np.median([out[n]["p1_med"] for n in ns])),4)}
out["BY_DOMAIN"]=by; out["gate"]=rate; out["head_formula"]="rms(h)@E^T, no norm weight (verified vs model logits)"
print(json.dumps(by,ensure_ascii=False))
json.dump(out,open("../results/e64/A7_margin.json","w"),ensure_ascii=False); print("saved")

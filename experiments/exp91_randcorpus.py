#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9-2 (audit2 H2): клетка RANDOM-ВЕСА × CORPUS teacher-forced.
Конфаунд «dip выучен»: прежний R-null = случайные веса × СВОЙ МУСОР (вырожденный вход)
→ dip=0 по построению оценщика. Разделяем: (случайные|обученные) × (корпус|перемешанный).
Оценщик ОДИН И ТОТ ЖЕ код-путь для всех четырёх клеток (curve_u + cos lag-1, L23, fp32,
output_hidden_states[23] — конвенция одна, включая trained — пересчитан заново, не npz).
Вердикт-правило: dip(train,corpus)≈−0.07; если dip(rand,corpus)≈dip(train,corpus) —
метрика измеряет ВХОД, а не веса («выучено» рушится); если ≈0 — заявка живёт.
САМОАУДИТ: from_config без дообучения (инициализация transformers, 2 сида); .eval();
токены = npz _tok (те же доки, что фичи); mask-гейт длины T>=64; float-касты в json.
"""
import json, os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, sys, torch
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp62_batt_lib import curve_u
from transformers import AutoModelForCausalLM
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L="23"; BASE="/data/Qwen/Qwen3.5-0.8B"
z=np.load(NPZ,allow_pickle=True)
meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=[n for n in sorted(meta) if dom[n]!="wiki"]
rngl=np.random.default_rng(7)
def dip_path(model,toks):
    with torch.no_grad():
        o=model(input_ids=torch.tensor([toks]),output_hidden_states=True)
    h=o.hidden_states[L][0,:len(toks)].double().numpy()
    hn=np.linalg.norm(h,axis=1,keepdims=True); u=h/np.maximum(hn,1e-12)
    P=curve_u(u,hn.ravel()); Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    c1=np.einsum("ij,ij->i",Pn[:-1],Pn[1:])
    return float(np.median(c1))
def cells(model,docs):
    out={"corpus":[],"shuffled":[]}
    for nm,toks in docs:
        out["corpus"].append(dip_path(model,toks))
        out["shuffled"].append(dip_path(model,list(rngl.permutation(toks))))
    return out
docs=[]
for nm in names:
    toks=np.asarray(z[f"{nm}_tok"]).ravel().astype(int).tolist()
    if len(toks)>=64: docs.append((nm,toks[:256]))
print(f"docs={len(docs)}",flush=True)
res={}
mt=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32).eval(); mt.requires_grad_(False)
res["trained"]=cells(mt,docs); print("trained готов",flush=True)
del mt
import gc; gc.collect()
for sd in [11,12]:
    torch.manual_seed(sd)
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(BASE)
    mr=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32).eval(); mr.requires_grad_(False)
    res[f"rand{sd}"]=cells(mr,docs); print(f"rand{sd} готов",flush=True)
    del mr; gc.collect()
tab={}
for k,v in res.items():
    tab[k]={m:[round(float(np.median(v[m])),4),
               round(float(np.percentile(v[m],2.5)),4),round(float(np.percentile(v[m],97.5)),4)] for m in v}
out=dict(n_docs=len(docs),table=tab,
         verdict=dict(train_corpus=tab["trained"]["corpus"][0],
                      rand_corpus_worst=max(abs(tab[f"rand{sd}"]["corpus"][0]) for sd in [11,12]),
                      null_band=[tab["trained"]["shuffled"][0][0],tab[f"rand11"]["shuffled"][0][0]]))
os.makedirs(ROOT+"/results/e92",exist_ok=True)
json.dump(out,open(ROOT+"/results/e92/R92_RANDCORPUS.json","w"),indent=1,ensure_ascii=False)
print(json.dumps(tab,indent=1,ensure_ascii=False))
print("R9-2 DONE",flush=True)

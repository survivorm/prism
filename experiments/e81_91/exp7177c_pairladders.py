#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-71b+E-77c ПАРНЫЕ ЛЕСТНИЦЫ (закрывает H7 и проверяет «ось шкалы»).
# Рецепт ПОЛНОСТЬЮ soliton-A: W = λ·trained + (1−λ)·rand(std-match, ndim≥2; 1D остаётся
# trained), randn seed 11. Пул — e77-й (4B-декод→0.8B ре-токен, 16 доков ≤192) —
# ПАРИРУЕТ лестницу E-71A (та была на 8 родных доках). Мера dip@L23 (curves_series
# дословно), медиана по 16.
#  A: все ndim≥2 (воспроизведение soliton-A на парном пуле). ПРЕ-РЕГ: λ≤0.4 |dip|<0.02
#     И λ≥0.6 |dip|>0.07 — иначе «порог α» зависит от пула (тоже результат).
#  B: группы: early=layers 0..11, late=12..22, all=воспроизведение. Readout L23 в
#     early/late НЕ масштабируется. ПРЕ-РЕГ: early пересекает −0.075 при λ*∈[0.4,0.7];
#     late: max|dip−base|<0.03 на всей решётке (поздняя шкала не несёт порога).
#  Самопроверка: λ=1.0 всех клеток == base e77b (−0.150 ± 1e-3).
import ast, json, time
import os; os.environ.setdefault("OMP_NUM_THREADS","8")
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(12)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results"/"e71"; RAW=OUT/"PAIRLADDERS_raw.txt"; OUT.mkdir(parents=True,exist_ok=True)
src=open(ROOT/"experiments"/"kaggle_e64gpu"/"e64_gpu.py").read()
fns={n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
ns={"np":np}; [exec(fns[w],ns) for w in ["vpath_units","dcurve","curves_series"]]
curves_series=ns["curves_series"]
BASE="/data/Qwen/Qwen3.5-0.8B"; LAY=23
pool=load_calibration_data("Qwen/Qwen3.5-4B",max_length=256,min_length=64,n_samples=16,domain="wiki")
tk0=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B"); tk=AutoTokenizer.from_pretrained(BASE)
xs=[tk(tk0.decode(d["input_ids"][0].tolist()[:48]).strip(),add_special_tokens=False,
     return_tensors="pt")["input_ids"][0][:192] for d in pool]
m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32,attn_implementation="eager").eval()
caps={}
h=m.model.layers[LAY].register_forward_hook(lambda mod,i,o: caps.__setitem__(LAY,o[0] if isinstance(o,tuple) else o))
sd0={k:v.clone() for k,v in m.state_dict().items()}
g=torch.Generator().manual_seed(11)
sd_r={k:(torch.randn(v.shape,generator=g)*v.std()) if v.ndim>=2 else v.clone() for k,v in sd0.items()}
def mix(mask):
    with torch.no_grad():
        for k,v in sd0.items():
            if v.ndim>=2 and mask(k): m.state_dict()[k].copy_(None*0 if False else k and m.state_dict()[k])
    # (реальный микс ниже — mask(k)? λ : 1)
def set_lambda(lam,mask):
    with torch.no_grad():
        for k,v in sd0.items():
            if v.ndim>=2 and mask(k):
                m.state_dict()[k].copy_(lam*v+(1-lam)*sd_r[k])
            else:
                m.state_dict()[k].copy_(v)
def dip_med():
    ds=[]
    for ids in xs:
        if ids.shape[0]<40: continue
        t=ids.unsqueeze(0); caps.clear()
        with torch.no_grad(): m.model(input_ids=t,use_cache=False)
        ds.append(float(curves_series(caps[LAY][0,:ids.shape[0]].to(torch.float32).numpy())["dip"]))
    return float(np.median(ds)),len(ds)
def lname_layer(k):
    p=k.split(".")
    return int(p[2]) if (len(p)>3 and p[1]=="layers") else -1
masks={"all":lambda k: True,
       "early":lambda k: 0<=lname_layer(k)<=11,
       "late":lambda k: 12<=lname_layer(k)<=22}
have=set()
if RAW.exists():
    for l in open(RAW): r=json.loads(l); have.add((r["grid"],r["group"],r["lam"]))
LAMS=[1.0,0.8,0.7,0.6,0.5,0.4,0.3]
for grid,groups in (("A",["all"]),("B",["early","late","all"])):
    for grp in groups:
        for lam in LAMS:
            if (grid,grp,lam) in have: continue
            t0=time.time(); set_lambda(lam,masks[grp])
            d,n=dip_med()
            with open(RAW,"a",encoding="utf-8") as f:
                f.write(json.dumps(dict(grid=grid,group=grp,lam=lam,dip=round(d,4),n=n,secs=round(time.time()-t0,1)))+"\n")
            print(f"[{grid} {grp} λ={lam}] dip {d:+.4f}",flush=True)
rows=[json.loads(l) for l in open(RAW)]
def get(g,gr,l): 
    r=[x for x in rows if x["grid"]==g and x["group"]==gr and x["lam"]==l]
    return r[0]["dip"] if r else None
print("[самопроверка λ=1]:", [get(g,"all",1.0) for g in ("A","B")], "(e77b base −0.150)",flush=True)
A=[get("A","all",l) for l in LAMS]
print(f"[пре-рег A] λ≤0.4: {A[-2:]} (<0.02?) λ≥0.6: {A[1:4]} (>0.07?)",flush=True)
E=[get("B","early",l) for l in LAMS]; Lt=[get("B","late",l) for l in LAMS]; base=A[0]
cross=[l for l,d in zip(LAMS,E) if d is not None and d<-0.075]
print(f"[пре-рег B] early пересек −0.075 при λ∈{cross} (ожидали ⊂[0.4,0.7]) | late max|Δ| {max(abs(x-base) for x in Lt if x is not None):.4f} (<0.03?)",flush=True)
print("PAIRLADDERS DONE",flush=True)

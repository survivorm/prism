#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76k ЗЕРКАЛО: «вторая компонента = свет, рассеянный readout-зеркалом обратно».
# Механика: в own-траектории шаг t движется из состояния, собранного из токена,
# ПРОПУЩЕННОГО зеркалом на шаге t−1 (p1[t] — уверенность readout, выбравшего tok[t];
# Δ_t — движение ПОСЛЕ вхождения tok[t] ⇒ сцепка e2(t)↔surp(t) есть обратная связь
# одного кругового пролёта; surp(t−1),(t−2) — многократные пролёты).
# Контроль: корпус той же модели — readout НИКОГО не feeding back (токены из текста).
# e2(t)=((Δ_t·u2)²/||Δ_t||²) в общем базисе R=16 (cov corpus∪own), u2 — 2-й собствен
# вектор КЛЕТКИ (где второй член определён); e1 — внутренний контроль-канал.
# ПРЕ-РЕГ: рассеяние ЖИВО если в own |ρ(e2,surp)| ≥ 0.10 и CI95(док-бут) не содержит 0
# ХОТЯ БЫ на одном из lag 0..2, И корпусной контроль строго слабее (неперекрывающиеся
# CI или знак противоположен). Иначе — «общая черта потока» / мертво.
import ast, json
import os; os.environ.setdefault("OMP_NUM_THREADS","8")
os.environ["PRISM_CALIB_PIN_DIR"]=str(__import__("pathlib").Path(__file__).resolve().parent.parent/"data"/"calib_pins")
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(10)
from transformers import AutoModelForCausalLM
from prism.calibration import load_calibration_data
ROOT=Path(__file__).resolve().parent.parent; D=ROOT/"results"/"e64"/"tpu"
R=16; L=23
rng=np.random.default_rng(61)
src=open(ROOT/"experiments"/"kaggle_e64gpu"/"e64_gpu.py").read()
fns={n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
ns={"np":np}; exec(fns["prompt_ids"],ns); prompt_ids=ns["prompt_ids"]
pool=load_calibration_data("Qwen/Qwen3.5-4B",max_length=256,min_length=64,n_samples=16,domain="wiki")
# корпусные p1 teacher-forced (одно полное прохождение на док, greedy-p1 = max softmax)
m=AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B",dtype=torch.float32,
                                       attn_implementation="eager").eval()
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
import json as js
pin=js.load(open(ROOT/"data"/"calib_pins"/"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"))
cor_p1=[]
for i in range(16):
    ids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"][:256]
    t=torch.tensor([ids])
    with torch.no_grad(): o=m(input_ids=t)
    p=torch.softmax(o.logits[0,:-1].float(),-1)
    cor_p1.append(p.gather(1,t[0,1:].unsqueeze(1)).squeeze(1).numpy().astype(np.float64))
print("[corpus p1 посчитаны]",[round(float(np.median(x)),2) for x in cor_p1[:4]],flush=True)
del m
z_o=np.load(D/"P1_own.npz"); z_c=np.load(D/"P2_corpus.npz"); sfx=f"_L{L}"
def rows(z):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
Uc=rows(z_c);Uo=rows(z_o)
Call=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(Call);B=V[:,-R:]
def u2_of(Us):
    Xs=[];Ys=[]
    for U in Us: A=U@B; Xs.append(A[:-1]);Ys.append(A[1:])
    Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x in Xs)
    M=np.linalg.solve(Sxx.T,Sxy.T).T;w,Vv=np.linalg.eigh(.5*(M+M.T))
    return Vv[:,np.argsort(w)[:2]]
def e_series(Us,plane):
    out=[]
    for U in Us:
        A=U@B; Dl=A[1:]-A[:-1]
        n2=np.sum(Dl*Dl,axis=1)+1e-12
        out.append(np.stack([np.mean((Dl@plane[:,k])**2)/0 if False else (Dl@plane[:,k])**2/n2 for k in range(2)],1))
    return out   # list [T,2]
def lagc(E,surp,k,ch):
    xs=[];ys=[]
    for e,s in zip(E,surp):
        Tm=min(len(e),len(s))
        e=e[:Tm];s=s[:Tm]
        idx=np.arange(Tm);idx=idx[idx>=8]
        if k>0: idx=idx[idx>=k]
        xs.append(e[idx,ch]);ys.append(s[idx-k if k>0 else idx])
    x=np.concatenate(xs);y=np.concatenate(ys)
    return float(np.corrcoef(x,y)[0,1])
def boot_doc(E,surp,k,ch):
    per=[]
    for e,s in zip(E,surp):
        Tm=min(len(e),len(s));e=e[:Tm];s=s[:Tm]
        idx=np.arange(Tm);idx=idx[idx>=max(8,k)]
        per.append((e[idx,ch],s[idx-(k if k>0 else 0)]))
    def tot(ii):
        x=np.concatenate([per[i][0] for i in ii]);y=np.concatenate([per[i][1] for i in ii])
        return float(np.corrcoef(x,y)[0,1])
    r0=tot(range(len(per)))
    bs=[tot(rng.integers(0,len(per),len(per))) for _ in range(300)]
    return r0,[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
Eo=e_series(Uo,u2_of(Uo));Ec=e_series(Uc,u2_of(Uc))
surp_o=[-np.log(np.clip(z_o[f"{k[:-len(sfx)]}_p1"].astype(np.float64),1e-6,1)) for k in sorted(z_o.files) if k.endswith(sfx)]
res={}
print(f"{'клетка':6s} {'канал':4s} "+" ".join(f"lag{k}" for k in range(3)),flush=True)
for nm,E,sp in (("own",Eo,surp_o),("cor",Ec,cor_p1)):
    for ch,cn in ((1,"e2"),(0,"e1")):
        line=[]
        res[f"{nm}_{cn}"]={}
        for k in (0,1,2):
            r,ci=boot_doc(E,sp,k,ch)
            res[f"{nm}_{cn}"][f"lag{k}"]=[round(r,3),[round(ci[0],3),round(ci[1],3)]]
            line.append(f"{r:+.3f}[{ci[0]:+.2f},{ci[1]:+.2f}]")
        print(f"{nm:6s} {cn:4s} "+" ".join(line),flush=True)
(D/"EIGEN_k.json").write_text(json.dumps(res,ensure_ascii=False,indent=1))
print("E76k DONE",flush=True)

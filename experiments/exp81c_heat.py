#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-81c HEAT (Рэндолл-Вилкинс наоборот): если послеcвечение = ловушки с распределением
глубин, «нагрев» (температура сэмплирования вместо greedy) обязан УСКОРЯТЬ релаксацию
(термический сход с ловушек) — ACF затухает быстрее, и/или сдвигается порог петель.
4 дока × {greedy, T=0.8} × 384 шага с пошаговым состоянием L23 (KV-кэш), проекция на
каноническую плоскость (базис из npz, тот же код). Меры: ACF e1 (лаги1..12),
slope ln|ACF| vs ln k (μ), rep8-доля и onset. Предсказание: μ(T) > μ(0) и/или
 persistence(T) < persistence(0); иначе — аналогия с ловушками шатается.
"""
import json, numpy as np, torch, os
from pathlib import Path
os.environ["PRISM_CALIB_PIN_DIR"]=str(Path(__file__).resolve().parent.parent/"data"/"calib_pins")
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e81"
BASE="/data/Qwen/Qwen3.5-0.8B"; N=384
tk=AutoTokenizer.from_pretrained(BASE)
m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32).eval(); m.requires_grad_(False)
pin=json.load(open(ROOT/"data"/"calib_pins"/"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"))
# плоскость из npz-канона
def rows(z,sfx="_L23"):
    o=[]
    for k in sorted(z.files):
        if k.endswith(sfx):
            H=z[k].astype(np.float64);H-=H.mean(0)
            o.append(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12))
    return o
zo=np.load(D/"P1_own.npz"); zc=np.load(D/"P2_corpus.npz")
Uo=rows(zo); Uc=rows(zc)
C=np.cov(np.concatenate(Uc+Uo,0).T);_,V=np.linalg.eigh(C);B=V[:,-2:]
Xs=[];Ys=[]
for U in Uo+Uc:
    A=U@B;Xs.append(A[:-1]);Ys.append(A[1:])
Mm=np.linalg.solve(sum(x.T@x for x in Xs).T,sum(x.T@y for x,y in zip(Xs,Ys)).T).T
w,Vv=np.linalg.eigh(.5*(Mm+Mm.T));pl=Vv[:,np.argsort(w)[:2]]
import torch.nn.functional as F
Bt=torch.tensor(B.T,dtype=torch.float32); Plt=torch.tensor(pl,dtype=torch.float32)
def gen(ids,T):
    hid=[]; toks=[]
    with torch.no_grad():
        o=m(torch.tensor([ids])); past=o.past_key_values
        # hidden_states не просили (o.hidden_states is None при output_hidden_states=False;
        # hasattr==True ловушка — ревью A/qwen Б3): состояния снимаем hook'ом ниже.
        # состояния — отдельным hook'ом
    hs={}
    def hk(mod,a,out2):
        y=out2[0] if isinstance(out2,tuple) else out2
        hs["h"]=y[0,-1].detach().float().numpy()
    han=m.model.norm.register_forward_hook(hk)
    ids_t=torch.tensor([ids]); 
    with torch.no_grad():
        o=m(ids_t); past=o.past_key_values
        lg=o.logits[0,-1].float()
        nxt=torch.argmax(lg) if T==0 else torch.multinomial(torch.softmax(lg/T,0),1)
        toks.append(int(nxt)); hid.append(hs["h"].copy())
        for _ in range(N-2):
            o=m(nxt.view(1,1),past_key_values=past,use_cache=True); past=o.past_key_values
            lg=o.logits[0,-1].float()
            nxt=torch.argmax(lg) if T==0 else torch.multinomial(torch.softmax(lg/T,0),1)
            toks.append(int(nxt)); hid.append(hs["h"].copy())
    han.remove()
    return np.array(hid),toks
def acf_slope(H):
    Hn=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-12)
    A=Hn@B; Dl=A[1:]-A[:-1]; Dl/=(np.linalg.norm(Dl,axis=1,keepdims=True)+1e-12)
    p=Dl@pl[:,0]; p=p-p.mean(); v=np.dot(p,p)+1e-12
    a=np.array([np.dot(p[:-(k)],p[k:])/v for k in range(1,13)])
    mask=np.abs(a)>=0.02; ks=np.arange(1,13)[mask]
    mu=None
    if mask.sum()>=5:
        beta,_,_,_=np.linalg.lstsq(np.stack([np.log(ks),np.ones(len(ks))]).T,np.log(np.abs(a[mask])),rcond=None)
        mu=float(beta[0])
    return a,mu
def rep_frac(toks,n=8):
    seen=set(); f=[]
    for t in range(len(toks)):
        g=tuple(toks[max(0,t-n+1):t+1])
        if len(g)<n: f.append(False); continue
        f.append(g in seen); seen.add(g)
    return float(np.mean(f[96:]))
res={}
for i in range(4):
    ids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"]
    row={}
    for T in (0.0,0.8):
        H,toks=gen(ids,T)
        a,mu=acf_slope(H)
        row[f"T{T}"]=dict(acf=[round(float(x),3) for x in a],mu=None if mu is None else round(mu,3),
                          rep8=round(rep_frac(toks),3))
        print(f"w{i} T={T}: ACF {np.round(a[:6],2).tolist()} mu {mu} rep8 {row[f'T{T}']['rep8']}",flush=True)
    res[f"w{i}"]=row
(OUT/"HEAT.json").write_text(json.dumps(res,indent=1,ensure_ascii=False,default=float))
print("E81C DONE",flush=True)

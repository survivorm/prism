#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-79 J-LENS MINI: проверка аналогии «наша общая λ₁-плоскость vs антропиковский
# J-space (привилегированное подпространство по усреднённому якобиану к логитам)».
# Оператор: J_L = E_{doc,pos}[ d logits / d h_L ] — скользящий градиент: h_L
# перехватывается forward-хуком как leaf, backward от (W·logits_row) при 32
# случайных ридоутах (JL-скетч), позиции — 16/док × 4 дока (wiki corpus TF,
# пул E-64 pin). Среднее J по позициям/докам → SVD, top-2 сингular-направления
# в пространстве h (1024) → проецируем в общий базис E-64 (cov corpus∪own) и
# сравниваем с нашей λ₁-плоскостью (R=16, E-76h-канон).
# ПРЕ-РЕГ: cos(mean-канон) > 0.5 на ≥2/3 слоёв ⇒ «тот же объект» — переформулировать
# режим функционально; < 0.3 ⇒ «показалось»; между — не разрешено (записать).
import ast, json
import os; os.environ.setdefault("OMP_NUM_THREADS","8")
os.environ["PRISM_CALIB_PIN_DIR"]=str(__import__("pathlib").Path(__file__).resolve().parent.parent/"data"/"calib_pins")
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(12)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
ROOT=Path(__file__).resolve().parent.parent; D=ROOT/"results"/"e64"/"tpu"
BASE="/data/Qwen/Qwen3.5-0.8B"; LAYERS=[1,11,23]; R=16; NJ=32
rng=np.random.default_rng(41)
tk=AutoTokenizer.from_pretrained(BASE)
docs=[d["input_ids"][0].tolist()[:192] for d in load_calibration_data(
    "Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=4,domain="wiki")]
m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32,attn_implementation="eager").eval()
m.requires_grad_(False)
capt={}
def hook(mod,i,o):
    x=o[0] if isinstance(o,tuple) else o
    x.retain_grad(); capt["h"]=x
hs=[m.model.layers[L].register_forward_hook(hook) for L in [max(LAYERS)]]  # перехват самого глубокого
def jac_at(layer,ids,pos):
    # forward с require_grad на входе слоя layer: делаем через pre-hook входного аргумента
    cap={}
    def pre(mod,args):
        a=args[0].detach().requires_grad_(True); cap["x"]=a; return (a,)
    hh=m.model.layers[layer].register_forward_pre_hook(pre)
    t=torch.tensor([ids])
    with torch.enable_grad():
        out=m(input_ids=t,use_cache=False)
        x=cap["x"][0]                          # [T,H] вход слоя layer
        # градиент скалярных ридоутов от logits на ПОЗИЦИИ pos — J строки
        W=rng.standard_normal((NJ, out.logits.shape[-1])).astype(np.float32)
        W=torch.tensor(W)
        r=(out.logits[0,pos]@W.T).sum()        # NJ ридоутов одним backward (сумма — не годится:
        g=torch.autograd.grad(r,x,retain_graph=False)[0][pos]   # даёт сумму ридоутов — ОДНО напр.
    hh.remove()
    return g.detach().numpy().astype(np.float64)   # [H] направление (случ. проекция выхода)
# Корректно: для подпространства входа достаточно 1 JL-направления выхода на позицию
# (ранг по входу оценивается по позициям, не по выходной размерности).
out={}
for L in LAYERS:
    Gs=[]
    for di,ids in enumerate(docs):
        for pos in rng.choice(len(ids)-2,16,replace=False)+1:
            Gs.append(jac_at(L,ids,int(pos)))
    G=np.stack(Gs)                       # [N,H] направления «на что влияет слой»
    G=G-G.mean(0); U_,S_,Vt=np.linalg.svd(G,full_matrices=False)
    Jdir=Vt[:2]                          # top-2 направления J-скобка
    # общий базис E-64 и наша плоскость (R=16)
    z=np.load(D/"P2_corpus.npz"); z2=np.load(D/"P1_own.npz"); sfx=f"_L{L}"
    Us=[]
    for zz in (z,z2):
        for k in sorted(zz.files):
            if k.endswith(sfx):
                Hh=zz[k].astype(np.float64);Hh-=Hh.mean(0)
                Us.append(Hh/(np.linalg.norm(Hh,axis=1,keepdims=True)+1e-12))
    Call=np.cov(np.concatenate(Us,0).T); _,V=np.linalg.eigh(Call); B=V[:,-R:]
    def planes(U): A=U@B; return A[:-1],A[1:]
    Xs=[];Ys=[]
    for U in Us[len(Us)//2:]:  # own-клетка
        x,y=planes(U); Xs.append(x);Ys.append(y)
    Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x in Xs)
    M=np.linalg.solve(Sxx.T,Sxy.T).T; w,Vv=np.linalg.eigh(.5*(M+M.T))
    P=Vv[:,np.argsort(w)[:2]]            # [R,2] в базисе B
    Jb=Jdir@B                            # [2,R] J-направления в базисе B
    Qb=np.linalg.qr(Jb.T)[0]             # [R,2]
    cs=np.clip(np.linalg.svd(P.T@Qb,compute_uv=False),0,1)
    out[f"L{L}"]=dict(cos2=[round(float(x),3) for x in cs],
        mean=round(float(cs.mean()),3),J_S=[round(float(x),1) for x in S_[:3]])
    print(f"L{L}: cos(наша λ₁-плоскость, top-2 J) = {out[f'L{L}']['cos2']} mean {out[f'L{L}']['mean']:.3f}",flush=True)
(D/"EIGEN_J.json").write_text(json.dumps(out,ensure_ascii=False,indent=1))
print("E79 DONE",flush=True)

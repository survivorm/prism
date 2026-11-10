#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76m ОПТИЧЕСКИЙ АТЛАС: каждый слой — полупрозрачное зеркало (beam splitter):
# h_out = h_in + f(h). Интенсивность на выходе: ‖h+f‖²=‖h‖²+‖f‖²+2⟨h,f⟩.
# На метр на позицию и слой (teacher-forcing, 16 wiki-доков, все 24 слоя, 1 forward/док):
#   gain  = ‖h_out‖/‖h_in‖
#   refl  = ‖f‖/‖h_in‖                      («отражательная способность»)
#   interf= cos(h_in, f)                     (знак интерференции: + собирательная, − гасящая)
# ПРЕ-РЕГ (из E-77b «ранние несут, поздние нет»): r_l убывает с глубиной ИЛИ поздние
# слои имеют малый вклад в dip; если atlas покажет поздние слои с БОЛЬШИМ refl —
# «ранне-тяжесть» E-77b требует пересмотра (поздние энергичны, но не несут dip —
# разделение «энергия ≠ носитель»). Антифазные (interf<0) слои — кандидаты в «гасящие
# зеркала», их профиль по глубине — новый факт.
import json
import os; os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(10)
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"results"/"e76"
OUT.mkdir(exist_ok=True)
pin=json.load(open(ROOT/"data"/"calib_pins"/"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"))
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
m=AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B",dtype=torch.float32,
                                       attn_implementation="eager").eval()
nL=len(m.model.layers)
rec={}
def mk(l):
    def hk(mod,args,out):
        x=args[0][0] if isinstance(args[0],tuple) else args[0]
        y=out[0] if isinstance(out,tuple) else out
        rec[l]=(x.detach(),y.detach())
    return hk
hs=[m.model.layers[l].register_forward_hook(mk(l)) for l in range(nL)]
G=np.zeros((nL,0));R=np.zeros((nL,0));I=np.zeros((nL,0))
gains=[];refls=[];inters=[]
with torch.no_grad():
    for i in range(16):
        ids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"][:224]
        m(input_ids=torch.tensor([ids]))
        g=np.zeros(nL);r=np.zeros(nL);it=np.zeros(nL)
        for l in range(nL):
            x,y=rec[l]; x=x.float().numpy(); y=y.float().numpy()
            f=y-x
            nx=np.linalg.norm(x,axis=1)+1e-12
            g[l]=np.median(np.linalg.norm(y,axis=1)/nx)
            r[l]=np.median(np.linalg.norm(f,axis=1)/nx)
            it[l]=np.median(np.einsum("ij,ij->i",x,f)/(nx*np.linalg.norm(f,axis=1)+1e-12))
        gains.append(g);refls.append(r);inters.append(it)
gains=np.array(gains);refls=np.array(refls);inters=np.array(inters)
atlas=dict(gain=np.median(gains,0).round(3).tolist(),
           refl=np.median(refls,0).round(3).tolist(),
           interf=np.median(inters,0).round(3).tolist(),
           refl_iqr=[np.percentile(refls,25,axis=0).round(3).tolist(),
                     np.percentile(refls,75,axis=0).round(3).tolist()])
(OUT/"OPTICS_atlas.json").write_text(json.dumps(atlas,ensure_ascii=False,indent=1))
print("слой: gain refl interf | refl межкварт.разброс")
for l in range(nL):
    print(f"L{l:2d}: {atlas['gain'][l]:.3f} {atlas['refl'][l]:.3f} {atlas['interf'][l]:+.3f} | "
          f"{atlas['refl_iqr'][0][l]:.3f}..{atlas['refl_iqr'][1][l]:.3f}",flush=True)
antiph=[l for l in range(nL) if atlas['interf'][l]<0]
print("антифазные зеркала (interf<0):",antiph,flush=True)
print("E76m DONE",flush=True)

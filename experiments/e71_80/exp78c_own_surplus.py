#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-78c Часть B: own-raw в ранжировании S (пре-рег части B был «после E-72-шаблона»;
# e72c: шаблон-эффекта нет; e75: qwen tmpl-сдвиг −0.019 ns ⇒ часть B легальна).
# Клетки: 16 own-greedy (P1_own.npz _tok, последние 384 генерированных; окно [32:288]
# — 256 токенов, «самообусловленность внутри окна» КАК У ДОКОВ; симметрия правил).
# Фон NG: wiki LOO БЕЗ дока-источника own_j (дисциплина контаминации e78-loop).
# dip_own: из P1_own_meta (L23, те же кривые батареи).
# ПРЕ-РЕГ: (a) медиана S_own < медиана S_docs ⇒ «низкоизбыточный режим» (как петля,
#   мягче); (b) ρ_s(S_own,depth_own) по 16: CI>0.2 и ρ≥0.5 ⇒ S драйвит и own;
#   всё остальное — не разрешено (линия S уже закрыта — это последний факт, не тест).
import ast, json, math
from collections import defaultdict
from pathlib import Path
import os; os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results"/"e78"
src=open(ROOT/"experiments"/"exp78_surplus.py").read()
segs={n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body
      if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
ns={"math":math,"defaultdict":defaultdict,"json":json,"Path":Path,"N":5}
exec(segs["load_pin_texts"].replace('PINS/f','ROOT/"data"/"calib_pins"/f'),ns)
exec(segs["NG"],ns)
NG=ns["NG"]; load_pin_texts=ns["load_pin_texts"]
tk=None
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
pools=[tk(t,add_special_tokens=False)["input_ids"][:256] for t in load_pin_texts("wiki")]
z=np.load(ROOT/"results"/"e64"/"tpu"/"P1_own.npz")
meta=json.load(open(ROOT/"results"/"e64"/"tpu"/"P1_own_meta.json"))
m=AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B",dtype=torch.float32,
                                       attn_implementation="eager").eval()
rows_old=json.load(open(OUT/"SURPLUS.json"))["rows"]
S_docs=np.array([r["S"] for r in rows_old if r["set"]=="wiki"])
D_docs=np.array([-r["dip"] for r in rows_old if r["set"]=="wiki"])
def r2s(r): pass
def spear(x,y):
    rx=np.argsort(np.argsort(x));ry=np.argsort(np.argsort(y))
    return float(np.corrcoef(rx,ry)[0,1])
S_own=[];Dp=[];res={}
B=32; W=256   # окно генерации [32:288]
for j in range(16):
    tok=z[f"own_w{j}_tok"][B:B+W]
    if len(tok)<128: S_own.append(None); continue
    t=torch.tensor([tok.tolist()])
    with torch.no_grad():
        o=m(input_ids=t)
        lg=o.logits[0,:-1].float(); tgt=t[0,1:]
        nll=float(torch.nn.functional.cross_entropy(lg,tgt))
    g=NG([d for i,d in enumerate(pools) if i!=j])
    S=round(nll-g.nll(tok.tolist()),4)
    d_own=-meta[f"own_w{j}"]["curves"]["23"]["dip"]
    S_own.append(S); Dp.append(-d_own if False else d_own)
    res[f"own_w{j}"]=dict(S=S,nll=round(nll,4),depth_own=round(d_own,4),n=len(tok))
S_own=np.array([s for s in S_own if s is not None]); Dp=np.array(Dp)
med_docs=float(np.median(S_docs)); med_own=float(np.median(S_own))
rng=np.random.default_rng(12)
bs=[spear(S_own[i],Dp[i]) for i in (rng.integers(0,16,16) for _ in range(1000))]
lo,hi=np.percentile(bs,[2.5,97.5]); rho=spear(S_own,Dp)
verdict=dict(med_S_docs=round(med_docs,3),med_S_own=round(med_own,3),
    a_low_surplus=bool(med_own<med_docs),
    rho_own=round(rho,3),ci95=[round(float(lo),3),round(float(hi),3)],
    b_driver=bool(rho>=0.5 and lo>0.2),
    note="часть B легальна после e72c/e75; окно генерации [32:288], фон LOO без источника")
(OUT/"SURPLUS_ownB.json").write_text(json.dumps(dict(cells=res,verdict=verdict),
                                                 ensure_ascii=False,indent=1))
print(json.dumps(verdict,ensure_ascii=False),flush=True); print("E78c DONE",flush=True)

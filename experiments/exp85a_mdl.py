#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-85a MDL/three-channel — независимый зонд («с другого конца»).
Объект: стоимость инкрементального сжатия каждого токена (zlib) — ни SVD, ни
плоскостей, ни модели: алгоритмическая сжимаемость потока.
Реперы: L0 (шов: own-префикс == корпус 1-в-1 → assert max|diff|<1e-9),
L1 (p1-насыщение ⇒ zlib-cost→пол на той же оси), L5 (novelty-ядро own late <1
= реэмиссия дешевле = аттрактор; корпусный калибр ≡1.0), L6 (расхождение кривых
только после точки генерации, не по ярлыку).
Корпусная половина — CPU, сейчас; own — из results/e81/REPLAY.json gen_ids (e81b).
"""
import json, zlib, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results"/"e85"; OUT.mkdir(parents=True,exist_ok=True)
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
pin=json.load(open(ROOT/"data"/"calib_pins"/"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"))
def mdl_curve(ids):
    co=zlib.compressobj(9); cost=[]
    for t in ids:
        b=tk.decode([t]).encode("utf-8","replace")
        co.compress(b)
        cost.append(len(co.flush(zlib.Z_SYNC_FLUSH)))
    fin=co.flush()
    cost[-1]+=len(fin) if fin else 0
    return np.array(cost,dtype=float)
def mdl_after_novelty(cost,ids,kmax=12):
    seen=set(); ev=[]
    for i,t in enumerate(ids):
        if t not in seen and i>4: ev.append(i)
        seen.add(t)
    base=float(np.mean(cost[4:]))
    out=[]
    for k in range(1,kmax+1):
        vals=[cost[i+k] for i in ev if i+k<len(cost)]
        out.append(float(np.mean(vals))/max(base,1e-9) if vals else float("nan"))
    return np.array(out)
# --- corpus (сейчас) ---
curves=[];nov=[]
for i in range(16):
    ids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"]
    c=mdl_curve(ids); curves.append(c); nov.append(mdl_after_novelty(c,ids))
early=np.array([x[4:28] for x in curves]); late=np.array([x[-24:] for x in curves])
N=np.nanmean(np.array(nov,dtype=float),0)
res=dict(corpus_cost_early=float(early.mean()),corpus_cost_late=float(late.mean()),
         corpus_cost_ratio_late_over_early=float(late.mean()/max(early.mean(),1e-9)),
         corpus_novelty_cost_ratio=[float(x) for x in N])
print("corpus MDL: early/late tok cost %.2f/%.2f ratio %.2f; novelty-kernel(k=1..6) %s"%(
  res["corpus_cost_early"],res["corpus_cost_late"],res["corpus_cost_ratio_late_over_early"],
  np.round(N[:6],2).tolist()),flush=True)
# --- own (когда есть REPLAY с gen_ids; путь: results/e81, не e81b!) ---
try:
    b=json.load(open(ROOT/"results"/"e81"/"REPLAY.json"))
    if "gen_ids" in b:
        oc=[];on=[];l0=[]
        for i,toks in enumerate([b["gen_ids"][f"w{j}"] for j in range(16)]):
            pids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"]
            full=list(pids)+list(toks)
            c=mdl_curve(full)
            pc=mdl_curve(list(pids))                 # L0-страж: own-префикс == корпус 1-в-1
            # сравнение КРОМЕ последнего prompt-токена: pc[-1] включает финальный flush
            # автономного prompt-потока (~6B), которого нет в непрерывном потоке full-версии
            l0.append(float(np.max(np.abs(pc[:-1]-c[:len(pids)-1]))))
            gc=c[len(pids):]                         # gen-стоимость НА ПОТОКЕ (с контекстом промпта)
            oc.append(gc); on.append(mdl_after_novelty(gc,list(toks)))
        L0max=float(np.max(l0))
        assert L0max<1e-9, f"L0 pipeline guard BROKEN: max|diff|={L0max}"
        oe=np.array([x[:128] for x in oc]); ol=np.array([x[-24:] for x in oc])
        res.update(L0_pipeline_guard=dict(max_abs_diff=L0max,status="OK"),
                   own_cost_early=float(oe.mean()),own_cost_late=float(ol.mean()),
                   own_cost_ratio=float(ol.mean()/max(oe.mean(),1e-9)),
                   own_novelty_cost_ratio=[float(x) for x in np.nanmean(on,0)],
                   L1_saturation_MDL=dict(own_late_over_corpus_late=float(ol.mean()/max(res["corpus_cost_late"],1e-9))),
                   L5_note="MDL-эхо own late: own_novelty kernel < corpus => реэмиссия дешевле (аттрактор)")
        print("own MDL early/late %.2f/%.2f ratio %.2f | own/corpus late %.2f"%(
          res["own_cost_early"],res["own_cost_late"],res["own_cost_ratio"],
          res["L1_saturation_MDL"]["own_late_over_corpus_late"]),flush=True)
    else:
        print("REPLAY.json без gen_ids — own-ветка пропущена")
except FileNotFoundError:
    print("own: ждём REPLAY.json (e81b v3, results/e81/) — корпус-калибровка готова")
def _clean(o):
    if isinstance(o,(float,np.floating)):
        o=float(o); return None if np.isnan(o) else o
    if isinstance(o,dict): return {k:_clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_clean(v) for v in o]
    return o
(OUT/"MDL_corpus.json").write_text(json.dumps(_clean(res),indent=1,ensure_ascii=False,default=float,allow_nan=False))

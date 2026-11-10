#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-78 РЕАНАЛИЗ (аудит shuffle-src): nll/dip из кэша RAW; NG — дословно из
# exp78_surplus.py (ast-сегменты, не копия). Шаг 1: перегон S для dom/loop — обязан
# совпасть с артифактом (формульная репродукция). Шаг 2: shuffle с ИСПРАВЛЕНЫМ
# фоном (исключается свой src, wiki[2] пропускался ⇒ прежний i!=j сдвигался).
import ast, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
src=open(ROOT/"experiments"/"exp78_surplus.py").read()
tree=ast.parse(src)
segs={n.name:ast.get_source_segment(src,n) for n in tree.body
      if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
ns={"math":math,"defaultdict":defaultdict,"json":json,"Path":Path}
exec(segs["load_pin_texts"].replace('PINS/f','__import__("pathlib").Path("data/calib_pins")/f'),ns)
exec(segs["NG"],ns)
NG=ns["NG"]; load_pin_texts=ns["load_pin_texts"]
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
DOMS=["wiki","math","code","chat"]
pools={d:[tk(t,add_special_tokens=False)["input_ids"][:256] for t in load_pin_texts(d)] for d in DOMS}
rng=np.random.default_rng(0)
wiki=pools["wiki"]; loop_src=min(3,len(wiki)-1)
loop_doc=(wiki[loop_src][:16]*14)[:224]
shuffle_docs=[]
for si0,d in enumerate(wiki[:4]):
    txt=tk.decode(d,skip_special_tokens=True)
    sents=[x for x in txt.split(". ") if x.strip()]
    if len(sents)>3:
        rng.shuffle(sents)
        ids=tk(". ".join(sents),add_special_tokens=False)["input_ids"][:256]
        shuffle_docs.append((si0,ids))
rows=[json.loads(l) for l in open(ROOT/"results"/"e78"/"SURPLUS_raw.txt")]
mism=0
for r in rows:
    if r["set"] in DOMS:
        g=NG([d for i,d in enumerate(pools[r["set"]]) if i!=r["doc"]]); ids=pools[r["set"]][r["doc"]]
    elif r["set"]=="loop":
        g=NG([d for i,d in enumerate(wiki) if i!=loop_src]); ids=loop_doc
    else:
        g=NG([d for i,d in enumerate(wiki) if i!=r["doc"]])   # СТАРЫЙ (смещённый) фон
        ids=shuffle_docs[r["doc"]][1]
    S=round(r["nll"]-g.nll(ids),4)
    if abs(S-r["S"])>1e-6:
        mism+=1; print("REPRO-MISMATCH",r["set"],r["doc"],S,r["S"])
print("формульная репродукция S:", "OK (68/68)" if mism==0 else f"{mism} рассогласований")
for r in rows:
    if r["set"]=="shuffle":
        src_i=shuffle_docs[r["doc"]][0]
        g=NG([d for i,d in enumerate(wiki) if i!=src_i])
        newS=round(r["nll"]-g.nll(shuffle_docs[r["doc"]][1]),4)
        print(f"shuffle{j={r['doc']}} src=wiki[{src_i}] S: {r['S']} → {newS}")
        r["S_fix"]=newS
Sd=np.array([r["S"] for r in rows if r["set"] in DOMS])
Dd=np.array([-r["dip"] for r in rows if r["set"] in DOMS])
sh_old=[r["S"] for r in rows if r["set"]=="shuffle"]; sh_new=[r.get("S_fix",r["S"]) for r in rows if r["set"]=="shuffle"]
lr=next(r for r in rows if r["set"]=="loop")
base_dip=float(np.median(Dd[[r["set"]=="wiki" for r in rows if r["set"] in DOMS]]))
print(f"mean shuffle S: {np.mean(sh_old):.3f} → {np.mean(sh_new):.3f} | медиана доков {np.median(Sd):.3f} | loop S {lr['S']} | base_dip {base_dip:.3f}")
verd=json.load(open(ROOT/"results"/"e78"/"SURPLUS.json"))["verdict"]
print("вердикт ρ-гейта не зависит от shuffle (домены):",verd["rho"],verd["ci95"],verd["verdict"],
      "| S-единый-канал:",verd["S_единый_канал"])

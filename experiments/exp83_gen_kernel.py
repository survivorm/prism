#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-83 GEN-KERNEL v3 (восстановлен 10.11; = v2 после аудита: без промпта в ядре,
собственный unigram-null, baseline v2 rows/null_rows, omc-парный CI (cor-CN)[:,k],
off-by-one 384 исправлен):
K(k)=P(x_{t+k}=x_t) по ГЕНЕРАЦИИ (без промпта — промпт это дословно корпус,
урок e81b), k=1..24, greedy 16×384 (тот же пин), сравнение с корпусным ядром
(_baseline, анти-эхо калибр) и СОБСТВЕННЫМ unigram-нулём перестановки.
Вердикт-гейт: sign-flip excess на k=2..6 (корпус минус, own плюс) = петля/эхо.
Ранняя/поздняя половина генерации — сегменты early/late (парный контраст в доке).
OMC-CI: own-медиана бустрапится по докам, корпусный контраст фиксирован ((cor-CN)[:,k]).
"""
import json, numpy as np, torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results"/"e83"; OUT.mkdir(exist_ok=True)
BASE="/data/Qwen/Qwen3.5-0.8B"; N=384; K=24
tk=AutoTokenizer.from_pretrained(BASE)
m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32).eval(); m.requires_grad_(False)
pin=json.load(open(ROOT/"data"/"calib_pins"/"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"))
def kernel(ids):
    return np.array([np.mean([ids[t]==ids[t+k] for t in range(len(ids)-k)]) if len(ids)>k else np.nan for k in range(1,K+1)])
def null_kernel(ids,rng,reps=3):
    return np.median([kernel(list(rng.permutation(ids))) for _ in range(reps)],0)
# ---- корпусный baseline v2 (rows + 2 null-пермута на док, порядок doc-major) ----
BP=OUT/"KERNEL_BASELINE.json"
if not BP.exists():
    rows=[];nullrows=[];rb=np.random.default_rng(99)
    for i in range(16):
        ids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"]
        rows.append(kernel(ids))
        for _ in range(2): nullrows.append(kernel(list(rb.permutation(ids))))
    json.dump(dict(rows=np.array(rows).tolist(),null_rows=np.array(nullrows).tolist()),open(BP,"w"),indent=1)
    print("baseline regenerated",flush=True)
b=json.load(open(BP))
cor=np.array(b["rows"]); corN=np.array(b["null_rows"])
CN=corN.reshape(16,2,K).mean(1)          # средний null дока
exc_c=cor-CN                             # попарно, 16×24
med_c=np.median(exc_c,0)
# ---- own генерации ----
gen_k=[];gen_null=[];seg_early=[];seg_late=[];gen_toks=[];RNG=np.random.default_rng(83)
for i in range(16):
    ids=tk(pin["texts"][i],add_special_tokens=False)["input_ids"]
    toks=[]
    with torch.no_grad():
        o=m(torch.tensor([ids])); past=o.past_key_values
        nxt=torch.argmax(o.logits[0,-1])
        toks.append(int(nxt))
        for _ in range(N-1):
            o=m(nxt.view(1,1),past_key_values=past,use_cache=True); past=o.past_key_values
            nxt=torch.argmax(o.logits[0,-1]); toks.append(int(nxt))
    gen_only=toks                        # БЕЗ промпта (промпт = дословно корпус, анти-эхо)
    gen_toks.append(gen_only)
    gen_k.append(kernel(gen_only))
    gen_null.append(null_kernel(gen_only,RNG))
    mlen=len(gen_only)//2
    seg_early.append(kernel(gen_only[:mlen])); seg_late.append(kernel(gen_only[mlen:]))
    print(f"w{i}: K_gen(1..6) {np.round(gen_k[-1][:6],4).tolist()}",flush=True)
G=np.array(gen_k); GN=np.array(gen_null)
# СВЕРКА С e81b: обе greedy на тех же пинах — генерации обязаны совпасть побайтово
try:
    bR=json.load(open(ROOT/"results"/"e81"/"REPLAY.json"))
    if "gen_ids" in bR:
        eq=[sum(a==b for a,b in zip(gt,bR["gen_ids"][f"w{j}"])) for j,gt in enumerate(gen_toks)]
        print("сверка e83↔e81b (совпавших токенов из 384 по докам):",eq[:8],"...",flush=True)
except FileNotFoundError:
    print("e81b REPLAY.json ещё нет — сверка отложена",flush=True)
exc=G-GN                                 # excess own-дока над ЕГО OWN unigram-нулём
med_g=np.median(exc,0)
ci=[]
for k in range(K):
    col=exc[:,k]; bs=[np.median(col[RNG.integers(0,16,16)]) for _ in range(2000)]
    ci.append([float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
out=dict(own_kernel=[float(x) for x in np.median(G,0)],
         own_excess=[float(x) for x in med_g],corpus_excess=[float(x) for x in med_c],
         own_boot_ci=ci,
         early_excess=[float(x) for x in np.median(np.array(seg_early)-GN,0)],
         late_excess=[float(x) for x in np.median(np.array(seg_late)-GN,0)])
print("\nown  excess:",np.round(med_g,4).tolist())
print("cor  excess:",np.round(med_c,4).tolist())
diff=med_g-med_c
cd=[]
for k in range(K):
    bs=[]
    for _ in range(2000):
        a=np.median(exc[RNG.integers(0,16,16),k]); bcol=(cor-CN)[:,k]
        bs.append(a-bcol)
    cd.append([float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
out["own_minus_corpus"]=[float(x) for x in diff]; out["omc_boot_ci"]=cd
print("own−cor      :",np.round(diff,4).tolist())
print("sig(k)       :",[int(l>0 or h<0) for l,h in cd],flush=True)
(OUT/"GEN_KERNEL.json").write_text(json.dumps(out,indent=1))
print("E83 DONE",flush=True)

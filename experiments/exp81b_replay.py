#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-81b REPLAY v3 (восстановлен 10.11 из состояния, проверенного смоук-тестом):
токено-уровневые повторы в own-генерациях (R3 с e81 отозван: p1 упёрся в
потолок 1.0 — Δp1 неизмерим; нужны настоящие n-граммы).
16 greedy-генераций (384 шага) по тем же вики-пинам, что E-64 P1_own.
ГЕЙТ воспроизведения: corr(сгенерированный p1-трек, сохранённый own_p1) > 0.99
для ≥12/16 доков, иначе артефакт помечается «replay-mismatch» и остальное не читается.
Меры: rep_frac(n) по ВСЕМУ контексту (prompt+gen, смещение off — урок e81b v1),
onset, profile сегментами 32; C) связь с хвостом e1 (TAIL16.json, единый lex-порядок);
СОХРАНЯЕТ gen_ids + p1-треки (для e85a MDL и e83-сверки).
"""
import json, numpy as np, torch, os
from pathlib import Path
os.environ["PRISM_CALIB_PIN_DIR"]=str(Path(__file__).resolve().parent.parent/"data"/"calib_pins")  # только для сверок с calibration; сам пин читается прямо ниже
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/"results"/"e64"/"tpu"; OUT=ROOT/"results"/"e81"; OUT.mkdir(parents=True,exist_ok=True)
BASE="/data/Qwen/Qwen3.5-0.8B"; N=384
tk=AutoTokenizer.from_pretrained(BASE)
m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32).eval()
m.requires_grad_(False)
pin=json.load(open(ROOT/"data"/"calib_pins"/"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"))
prompts=[tk(pin["texts"][i],add_special_tokens=False)["input_ids"] for i in range(16)]
zo=np.load(D/"P1_own.npz")
def rep_flags(ids,n):
    seen=set(); out=[]
    for t in range(len(ids)):
        g=tuple(ids[max(0,t-n+1):t+1])
        if len(g)<n: out.append(False); continue
        out.append(g in seen); seen.add(g)
    return np.array(out)
def tail16_of():  # (tail, keys) в ОДНОМ порядке: lex_order из файла
    try:
        j=json.load(open(OUT/"TAIL16.json"))
        return np.array(j["tail"]),[f"w{w}" for w in j["lex_order"]]
    except Exception: return None,None
def gen_one(ids):
    ids=torch.tensor([ids]); past=None; out_ids=[]; out_p1=[]  # ids=prompts[i] здесь
    with torch.no_grad():
        o=m(ids); past=o.past_key_values
        nxt=torch.argmax(o.logits[0,-1],keepdim=True).view(1,1)
        lp=float(torch.softmax(o.logits[0,-1].float(),-1)[int(nxt)].item())   # 1D logits: gather(1,..) невозможен
        out_ids.append(int(nxt)); out_p1.append(lp)
        for _ in range(N-1):
            o=m(nxt,past_key_values=past,use_cache=True); past=o.past_key_values
            lg=o.logits[0,-1].float()
            nxt=torch.argmax(lg,keepdim=True).view(1,1)
            out_ids.append(int(nxt)); out_p1.append(float(torch.softmax(lg,-1)[int(nxt)].item()))
    return out_ids,np.array(out_p1)
res={"gate":{},"docs":{}}
tails,tail_keys=tail16_of()
match=0
for i in range(16):
    gen,p1g=gen_one(prompts[i])
    ids=prompts[i]+gen                      # повтор считаем относительно ВСЕГО контекста (prompt+gen)
    off=len(prompts[i])
    p1s=zo[f"own_w{i}_p1"].astype(np.float64)   # СОХРАНЕНЫ КАК ВЕРОЯТНОСТИ p1 (max 1.0, 91%>0.9; ревью 10.11 B1: exp(-p) переворачивал гейт в −0.997)
    T=min(len(p1g),len(p1s))
    if len(p1g)!=len(p1s): print(f"w{i}: ВНИМАНИЕ длины треков {len(p1g)} vs {len(p1s)} — сравниваю первые {T}",flush=True)
    r=float(np.corrcoef(p1g[:T],p1s[:T])[0,1]) if T>20 and np.std(p1g[:T])>0 else float("nan")
    if np.isfinite(r) and r>0.99: match+=1
    reps={n:rep_flags(ids,n) for n in (4,8,16)}
    prof={n:[float(reps[n][off+a:off+a+32].mean()) for a in range(0,N-31,32)] for n in (4,8,16)}
    onsets={}
    for n in (4,8,16):
        rs=reps[n][off:]
        w=np.convolve(rs,np.ones(16)/16,mode="same")
        idx=np.where(w>0.25)[0]
        onsets[n]=int(idx[0]) if len(idx) else None
    res["gate"][f"w{i}"]=round(r,4) if np.isfinite(r) else None
    res["docs"][f"w{i}"]=dict(segments=prof,onset=onsets,p1_med=float(np.median(p1g)))
    res.setdefault("gen_ids",{})[f"w{i}"]=[int(x) for x in gen]
    res.setdefault("gen_p1",{})[f"w{i}"]=[float(x) for x in p1g]
    print(f"w{i}: gate r={r:.3f} rep8 med {np.median(reps[8][off:]):.2f} onset8 {onsets[8]} p1 {np.median(p1g):.3f}",flush=True)
res["gate_pass"]=f"{match}/16"
if match<12: res["replay_mismatch"]=True   # обещание докстринга: артефакт помечен, C/прочее не читается
if match>=12 and tails is not None:
    keys=[k for k in (tail_keys if tail_keys else sorted(res["docs"])) if k in res["docs"]]
    if tail_keys is None or len(keys)!=len(tails):
        print("C: TAIL16.json отсутствует/порядок не совпал — C-часть пропущена (нужен перепуск e81)",flush=True)
    else:
        rf8=np.array([float(np.mean(res["docs"][k]["segments"][8])) for k in keys])
        def spear(a,b):
            ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b))
            return float(np.corrcoef(ra,rb)[0,1])
        rng=np.random.default_rng(8181)
        obs=spear(tails,rf8)
        nl=[spear(rng.permutation(tails),rf8) for _ in range(2000)]
        res["C_tail_vs_rep8"]=dict(rho=round(obs,3),null=[round(float(np.percentile(nl,2.5)),3),round(float(np.percentile(nl,97.5)),3)])
        print(f"C: tail vs rep8 rho {obs:+.3f} null[{np.percentile(nl,2.5):+.3f},{np.percentile(nl,97.5):+.3f}]",flush=True)
(OUT/"REPLAY.json").write_text(json.dumps(res,indent=1,ensure_ascii=False,default=float))
print("E81B DONE",flush=True)

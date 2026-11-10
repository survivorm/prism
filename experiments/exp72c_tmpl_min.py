#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-72c TMPL-MIN (замена снятого матричного E-72, трек estimates/…_e72_cut_decision.md):
# единственный вопрос — убивает ли ШАБЛОН лупы у выровненной 0.8B (инстукт вне конверта).
# 12 промптов (6 chat + 6 math) × {raw, tmpl(собственный шаблон модели, guard на роли)} ×
# greedy 64. Пре-рег: loopy(tmpl) ≤ loopy(raw)−0.25 ⇒ шаблон спасает; |Δ|<0.15 ⇒ нет эффекта;
# обратный знак ⇒ вредит. Провенанс: sha текста, sha генерации, OOV, tmpl_src. Resume по паре.
import hashlib, json, time
from pathlib import Path
import numpy as np, torch
import os
os.environ.setdefault("OMP_NUM_THREADS","8")
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results"/"e72c"; OUT.mkdir(parents=True,exist_ok=True)
RAWF=OUT/"TMPL_MIN.jsonl"; BASE="/data/Qwen/Qwen3.5-0.8B"; GEN=64
P=["What is the capital of Australia?","Why is the sky blue?","How do I boil an egg?",
   "What is 17 times 4?","What is 144 divided by 12?","Solve: 3x = 21. What is x?",
   "Who wrote Hamlet?","What is 2 to the power 10?","Name two oceans.",
   "Compute 23 plus 19 minus 7.","What is the boiling point of water?","What is 7 factorial?"]
def rep4(s): 
    g=[tuple(s[i:i+4]) for i in range(len(s)-3)]
    return float(1-len(set(g))/max(1,len(g))) if g else 0.0
def main():
    done=set()
    if RAWF.exists():
        for l in open(RAWF):
            try: r=json.loads(l); done.add((r["form"], r["prompt_sha16"]))
            except Exception: pass
    tk=AutoTokenizer.from_pretrained(BASE)
    has_roles = ("user" in tk.chat_template and "assistant" in tk.chat_template)
    assert has_roles, "шаблон 0.8B не ролевой — тест бессмысленен"
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32,attn_implementation="eager").eval()
    vocab=set(tk.get_vocab().keys())
    for p in P:
        ph=hashlib.sha256(p.encode()).hexdigest()[:16]
        for form in ("raw","tmpl"):
            if (form,ph) in done: continue
            txt = tk.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,tokenize=False) if form=="tmpl" else p
            ids=tk(txt,add_special_tokens=False,return_tensors="pt")["input_ids"]
            t0=time.time()
            with torch.no_grad():
                out=m.generate(ids,max_new_tokens=GEN,do_sample=False,pad_token_id=tk.eos_token_id or 0)
            gen=out[0,ids.shape[1]:][:GEN].tolist()
            text=tk.decode(gen,skip_special_tokens=True)
            toks=tk.convert_ids_to_tokens(gen)
            oov=float(np.mean([("<unk" in t) or (t not in vocab) for t in toks]))
            qw=[w for w in p.lower().split() if len(w)>3]
            rel=float(np.mean([w in text.lower() for w in qw[:6]])) if qw else None
            row=dict(form=form,prompt_sha16=ph,prompt=p[:60],gen_sha16=hashlib.sha256(text.encode()).hexdigest()[:16],
                     rep4=rep4(gen),uniq=len(set(text.lower().split())),words=len(text.split()),oov=round(oov,3),
                     rel=rel,secs=round(time.time()-t0,1),text=text[:300])
            with open(RAWF,"a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
            print(f"{form}/{p[:22]} rep4 {row['rep4']:.2f} words {row['words']} rel {rel}", flush=True)
    rows=[json.loads(l) for l in open(RAWF)]
    lo=lambda fm: float(np.mean([r["rep4"]>0.45 for r in rows if r["form"]==fm])) if any(r["form"]==fm for r in rows) else None
    L_raw,L_tmpl=lo("raw"),lo("tmpl")
    verd=("шаблон спасает" if (L_tmpl is not None and L_raw-L_tmpl>=0.25) else
          "вредит" if (L_tmpl is not None and L_tmpl-L_raw>=0.25) else
          "нет эффекта" if L_tmpl is not None else "недописано")
    res=dict(loopy_raw=L_raw,loopy_tmpl=L_tmpl,verdict=verd,n=len(rows))
    (OUT/"TMPL_MIN.json").write_text(json.dumps(res,ensure_ascii=False,indent=1))
    print("E72C DONE",res,flush=True)
main()

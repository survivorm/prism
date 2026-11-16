# E-72 MODEL SCREEN: на какой модели greedy-генерация осмысленна (замечание пользователя:
# на 0.8B мы ловим битые ответы — математику можно на 0.8, проверять надо на «говорящей»).
# Кандидаты (локальные): Qwen3.5-4B (base), LFM3-1.7B, Qwen3-VL-4B-Instruct (текст через шаблон).
# Классы вопросов: wiki-продолжения (16, пул), бытовые вопросы (12), арифметика/факты (12, OpenMath).
# Метрики: p_same, rep16, self-4gram-repetition (доля повторов 4-грамм внутри текста), loopy-ставка,
# обрыв (no-eos), примеры текстов в md. Порог «говорящей» модели: loopy≤25% И rep4≤0.25 по классам.
import json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(10)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e72"; os.makedirs(OUT,exist_ok=True)
GEN=64
wiki=[d for d in load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")]
wiki_prompts=[t["text"] if isinstance(t,dict) and "text" in t else "" for t in wiki]
# text может отсутствовать — соберём из пула декодом
tk0=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
wiki_prompts=[tk0.decode(t["input_ids"][0].tolist()[:48]) for t in wiki]
chat=["What is the capital of Australia?","Why is the sky blue?","Name three primary colors.",
      "How do I boil an egg?","What is gravity in one sentence?","Explain what a computer is briefly.",
      "Who wrote Hamlet?","What is the boiling point of water?","Name two oceans.",
      "Why do we sleep?","What is a verb?","How tall is a giraffe roughly?"]
math=["What is 17 times 4?","If a train travels 60 km in 40 minutes, what is its speed in km/h?",
      "What is 144 divided by 12?","Compute 23 plus 19 minus 7.","What is 15% of 200?",
      "Solve: 3x = 21. What is x?","What is the square root of 81?","What is 7 factorial?",
      "A rectangle is 5 by 8, what is its area?","What is 2 to the power 10?",
      "If 5 pens cost 3 dollars, how much do 10 pens cost?","What is 1000 minus 387?"]
SETS={"wiki":wiki_prompts,"chat":chat,"math":math}
def rep4(s_ids):
    g=[tuple(s_ids[i:i+4]) for i in range(len(s_ids)-3)]
    return float(1-len(set(g))/max(1,len(g))) if g else 0.0
res={"models":{}}
FN=f"{OUT}/SCREEN.json"
if os.path.exists(FN): res.update(json.load(open(FN)))
def save(): json.dump(res,open(FN,"w"),default=float)
CANDS=[("qwen35_4b","Qwen/Qwen3.5-4B",False),
       ("lfm3_1.7b","LiquidAI/LFM3-1.7B",True),
       ("qwen3vl_4b_inst","Qwen/Qwen3-VL-4B-Instruct",True)]
for tag,mid,is_chat in CANDS:
    if tag in res["models"]: print("skip",tag,flush=True); continue
    try:
        tk=AutoTokenizer.from_pretrained(mid,trust_remote_code=True)
        m=AutoModelForCausalLM.from_pretrained(mid,trust_remote_code=True,
            dtype=torch.float32,attn_implementation="eager").eval()
    except Exception as e:
        print("!! load",tag,e,flush=True); res["models"][tag]=dict(error=str(e)); save(); continue
    per={}
    for st,prompts in SETS.items():
        t0=time.time(); rows=[]
        for pi,p in enumerate(prompts):
            try:
                if is_chat:
                    try: txt=tk.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,tokenize=False)
                    except Exception: txt="USER: "+p+"\nASSISTANT:"
                else: txt=p
                ids=tk(txt,add_special_tokens=False,return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    out=m.generate(ids,max_new_tokens=GEN,do_sample=False,
                        pad_token_id=tk.eos_token_id if tk.eos_token_id is not None else 0)
                gen=out[0][ids.shape[1]:].tolist()
                gen=gen[:GEN]
                row=dict(n=len(gen),p_same=float(np.mean([gen[i]==gen[i-1] for i in range(1,len(gen))])),
                    rep16=float(np.mean([gen[i]==gen[i-16] for i in range(16,len(gen))])) if len(gen)>20 else None,
                    rep4=rep4(gen),text=tk.decode(gen)[:220])
                rows.append(row)
            except Exception as e:
                rows.append(dict(error=str(e)[:160]))
        ok=[r for r in rows if "error" not in r]
        per[st]=dict(n=len(ok),n_err=len(rows)-len(ok),
            p_same=round(float(np.median([r["p_same"] for r in ok])),3) if ok else None,
            rep4=round(float(np.median([r["rep4"] for r in ok])),3) if ok else None,
            loopy_frac=round(float(np.mean([r["rep4"]>0.45 for r in ok])),2) if ok else None,
            cut_frac=round(float(np.mean([r["n"]>=GEN-1 for r in ok])),2) if ok else None,
            samples=[r["text"][:150] for r in ok[:3]])
        res["models"][tag]=per; save()
        print(f"[{time.time()-t0:.0f}s] {tag}/{st} p_same {per[st]['p_same']} rep4 {per[st]['rep4']} loopy {per[st]['loopy_frac']} err {per[st]['n_err']}",flush=True)
    del m; import gc; gc.collect()
print("E72 DONE",flush=True)

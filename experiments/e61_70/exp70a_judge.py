# E-70a COHERENCE JUDGE: адекватность текстов, а не только цифры.
# Тексты E-68 (v1) и E-68b (v2) greedy/harness (8 wiki-промптов × 64 шага) судятся Qwen3.5-4B (CPU):
#  nll_cond   = NLL продолжения под промптом (связность с темой, кросс-модельный судья)
#  nll_uncond = NLL продолжения без промпта (внутренняя связность текста)
#  relevance  = nll_uncond − nll_cond  (>0: текст про то, о чём промпт)
# плюс РАЗВЕРНУТЫЕ тексты в человекочитаемый md — глаз остаётся финальным арбитром (замечание
# пользователя: «цифры не всегда решают»). Судья — та же фамилия моделей: предвзятость признана,
# глазом проверяется.
import json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(6)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e70"; os.makedirs(OUT,exist_ok=True)
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
prompts={i:d["input_ids"][0].tolist()[:64] for i,d in enumerate(pool[:8])}
J=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B".replace("0.8B","4B"),dtype=torch.float32).eval()
def nll_of(ids,ctx=None):
    t=torch.tensor([ (ctx or [])+ids ])
    P=len(ctx or [])
    with torch.no_grad(): o=J(input_ids=t,use_cache=False)
    lg=o.logits[0,P-1:len(ids)+P-1].float()   # предсказания на позициях ids
    lb=torch.tensor(ids)
    return float(torch.nn.functional.cross_entropy(lg,lb))
res={}
t0=time.time()
for run,path in [("v1","results/e68/HARNESS.json"),("v2","results/e68b/HARNESS.json")]:
    d=json.load(open(path))
    for k in sorted(d["docs"],key=lambda x:int(x.split("_")[1])):
        di=int(k.split("_")[1]); row={}
        for arm in ["greedy","harness"]:
            toks=d["docs"][k][arm]["toks"]
            c=nll_of(toks,prompts[di]); u=nll_of(toks)
            row[arm]=dict(nll_cond=round(c,3),nll_uncond=round(u,3),relevance=round(u-c,3),
                          text=tk.decode(toks))
            print(f"[{time.time()-t0:5.0f}s] {run}/{k}/{arm} cond {c:.2f} uncond {u:.2f} rel {u-c:+.2f}",flush=True)
        res[f"{run}/{k}"]=row
        json.dump(res,open(f"{OUT}/JUDGE_E68.json","w"),indent=1,ensure_ascii=False)
# сводка + человекочитаемая панель
import statistics as st
summ={}
for arm in ["greedy","harness"]:
    for run in ["v1","v2"]:
        rows=[res[f"{run}/{k}"][arm] for k in res if k.startswith(run)]
        summ[f"{run}/{arm}"]=dict(med_cond=round(st.median(r["nll_cond"] for r in rows),3),
            med_rel=round(st.median(r["relevance"] for r in rows),3))
res["summary"]=summ; json.dump(res,open(f"{OUT}/JUDGE_E68.json","w"),indent=1,ensure_ascii=False)
with open("docs/dynamic/coherence_panel.md","w") as f:
    f.write("# Панель адекватности (судья 4B + глаз)\n\nСводка: `"+json.dumps(summ,ensure_ascii=False)+"`\n\n")
    for run in ["v1","v2"]:
        for k in sorted([x for x in res if x.startswith(run)],key=lambda x:int(x.split("/")[1].split("_")[1])):
            f.write(f"\n## {run} {k.split('/')[1]}  (greedy rel {res[k]['greedy']['relevance']:+.2f} | harness rel {res[k]['harness']['relevance']:+.2f})\n")
            for arm in ["greedy","harness"]:
                f.write(f"- **{arm}**: {res[k][arm]['text'][:400]}\n")
print(json.dumps(summ,indent=1,ensure_ascii=False),flush=True); print("E70A DONE",flush=True)

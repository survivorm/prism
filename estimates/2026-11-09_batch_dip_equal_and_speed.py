#!/usr/bin/env python3
# ПРИКИДКА→РЕШЕНИЕ: дип-метрику можно батчить ТОЛЬКО если right-pad точен для ГИБРИДА
# (attention каузален ⇒ точен; gated-deltanet — сканер, под вопросом). Тест равенства —
# по самой метрике (dip L23), не по скрытым состояниям «на глаз». Скорость — отношение при
# конкуренции с e72 (консенсус: отношения валидны при ±порядок). Решение: E77_BATCH=1
# только если rel-dip-различие < 1e-3 И ускорение ≥ 2.
import os; os.environ["OMP_NUM_THREADS"]="8"; os.environ["TOKENIZERS_PARALLELISM"]="false"
import ast, time, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT="/home/survivor/work/prism"
src=open(f"{ROOT}/experiments/kaggle_e64gpu/e64_gpu.py").read()
fns={n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
ns={"np":np}
for w in ["vpath_units","dcurve","curves_series"]: exec(fns[w],ns)
cs=ns["curves_series"]
tk=AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
m=AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B",dtype=torch.float32,attn_implementation="eager").eval()
caps={}
hs=[m.model.layers[23].register_forward_hook(lambda mo,i,o: caps.__setitem__('h',o.detach()))]
txt=("The history of the city began with a market. Roman sources describe the settlement as a "
    "trading post on the river, where salt and grain changed hands. ")
docs=[tk(txt+str(i), add_special_tokens=False)["input_ids"][:128] for i in range(8)]
def dip_h(h): return float(cs(h.to(torch.float32).numpy())["dip"])
t0=time.time()
with torch.no_grad():
    seq=[dip_h(m(input_ids=torch.tensor([x])).logits.new_empty(0).new_tensor([]) if False else caps and (m.model(input_ids=torch.tensor([x])) or caps['h'][0,:len(x)])) for x in docs]
    # (m.model(...) возвращает объект; caps заполнен хуком — выражение выше: forward then slice)
t_seq=time.time()-t0
mx=max(len(x) for x in docs); P=torch.zeros(len(docs),mx,dtype=torch.long)
for i,x in enumerate(docs): P[i,:len(x)]=torch.tensor(x)
t0=time.time()
with torch.no_grad(): m.model(input_ids=P)
t_bat=time.time()-t0
H=caps['h']
bat=[dip_h(H[i,:len(x)]) for i,x in enumerate(docs)]
d=max(abs(a-b) for a,b in zip(seq,bat))
print(f"dip seq: {np.round(seq,4)}")
print(f"dip bat: {np.round(bat,4)}")
print(f"max |Δdip| = {d:.2e} (гейт 1e-3) | скорость seq {t_seq:.1f}s vs batch {t_bat:.1f}s → ×{t_seq/t_bat:.2f}")
print("РЕШЕНИЕ:", "батчить" if (d<1e-3 and t_seq/t_bat>=2) else "оставить последовательно")
for h in hs: h.remove()

# lfm3_gen.py — генерация текстов LFM3-1.7B (greedy, 256 токенов) на wiki-пинах для клетки
# «чужие токены» матрицы чтения. Выход: LFM3_TEXT.json {doc: text} —下游 TPU-ядро читает его
# через kernel_sources и ре-токенизирует Qwen'ом (сравнение на уровне ТЕКСТА).
import subprocess, sys, os, json, time
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2"])
import numpy as np, torch
IN="/kaggle/input"; OUT="/kaggle/working"
def find(name):
    for root,dirs,files in os.walk(IN):
        if name in files: return os.path.join(root,name)
        if name in dirs: return os.path.join(root,name)
    raise FileNotFoundError(name)
P=find("prism-calib-pins-qwen")
pin=None
for root,dirs,files in os.walk(P):
    for f in files:
        if f.startswith("wiki___data"): pin=json.load(open(os.path.join(root,f)))
assert pin
from transformers import AutoModelForCausalLM, AutoTokenizer
m=AutoModelForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B",trust_remote_code=True,dtype=torch.bfloat16,
    attn_implementation="sdpa").to("cuda").eval()
tk=AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B",trust_remote_code=True)
texts=pin["texts"][:64]
out={}
for i,t in enumerate(texts):
    ids=tk(t,add_special_tokens=False)["input_ids"][:64]
    x=torch.tensor([ids],dtype=torch.long,device="cuda")
    with torch.no_grad():
        g=m.generate(x,max_new_tokens=256,do_sample=False,
                     attention_mask=torch.ones_like(x),use_cache=True)
    gen=g[0,64:].tolist()
    out[f"w{i}"]=tk.decode(gen)
    if i%8==7: print(i,"%.0fs"%(time.time()-t0),flush=True)
json.dump(out,open(f"{OUT}/LFM3_TEXT.json","w"),ensure_ascii=False,indent=0)
# токен-статы для сверки с lfm3_pool-эпохой
rep16=float(np.mean([len(set(tk.encode(out[k]))[i:i+16])<12 for k in out for i in range(0,200,16)]))
print("saved; rep16~%.3f n=%d"%(rep16,len(out)),flush=True)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)

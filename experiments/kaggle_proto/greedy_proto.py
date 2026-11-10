# greedy_proto.py — вскрывающая проба: почему траектория не сходится с эталонным пулом.
# 4 клетки на wiki_0 (256 шагов) + 1 на HF-весах. Метрики: p_same, gen_sha, dip, префикс.
import subprocess, sys, os, json, time, hashlib
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2","datasets==5.0.1"])
import os as _o; _o.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
IN="/kaggle/input"; OUT="/kaggle/working"
def sha(x): return hashlib.sha256(repr(x).encode()).hexdigest()[:12]
def find(name):
    for root,dirs,files in os.walk(IN):
        if name in files: return os.path.join(root,name)
        if name in dirs: return os.path.join(root,name)
    raise FileNotFoundError(name)
W=find("qwen35-08b-weights"); P=find("prism-calib-pins-qwen")
pin=None
for root,dirs,files in os.walk(P):
    for f in files:
        if f.startswith("wiki___data"): pin=json.load(open(os.path.join(root,f)))
ids=tk=None
os.makedirs("/data/Qwen",exist_ok=True); tgt="/data/Qwen/Qwen3.5-0.8B"
if not os.path.exists(tgt): os.symlink(W,tgt)
m=AutoModelForCausalLM.from_pretrained(tgt,dtype=torch.float32).to("cuda").eval()
tk=AutoTokenizer.from_pretrained(tgt)
toks0=tk(pin["texts"][0],add_special_tokens=False)["input_ids"][:64]
REF=json.load(open(os.path.join(P,"exp47b_gen_kinematics.json"))) if os.path.exists(os.path.join(P,"exp47b_gen_kinematics.json")) else None
ref_g={f"{d['dom']}_{d['si']}":d["gen_sha"] for d in REF["docs"]} if REF else {}
print("ref sha wiki_0:",ref_g.get("wiki_0"),flush=True)

def stats(gen):
    g=np.array(gen)
    return dict(p_same=round(float(np.mean(g[1:]==g[:-1])),3),sha=sha(gen),
                uniq=round(float(len(set(gen))/len(gen)),3),head=tk.decode(gen[:18])[:90])

def gen_dup(model,ids,steps):
    # v1-семантика: шаг = forward(последний токен) для логов, затем forward(nxt) для состояния
    c=DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=torch.tensor([ids],device="cuda"),past_key_values=c,use_cache=True)
        toks=list(ids); gen=[]
        for j in range(steps):
            lg=model(input_ids=torch.tensor([[toks[-1]]],device="cuda"),past_key_values=c,use_cache=True).logits[0,-1]
            nx=int(lg.argmax()); toks.append(nx); gen.append(nx)
            model(input_ids=torch.tensor([[nx]],device="cuda"),past_key_values=c,use_cache=True)
    return gen

def gen_clean(model,ids,steps):
    c=DynamicCache(config=model.config)
    with torch.no_grad():
        o=model(input_ids=torch.tensor([ids],device="cuda"),past_key_values=c,use_cache=True)
        lg=o.logits[0,-1]; toks=list(ids); gen=[]
        for j in range(steps):
            nx=int(lg.argmax()); toks.append(nx); gen.append(nx)
            o=model(input_ids=torch.tensor([[nx]],device="cuda"),past_key_values=c,use_cache=True)
            lg=o.logits[0,-1]
    return gen

def gen_padded(model,ids,steps,maxlen=64+256+32):
    buf=torch.zeros((1,maxlen),dtype=torch.long,device="cuda")
    mask=torch.zeros((1,maxlen),dtype=torch.long,device="cuda")
    buf[0,:64]=torch.tensor(ids,device="cuda"); mask[0,:64]=1
    toks=list(ids); gen=[]
    with torch.no_grad():
        lg=model(input_ids=buf,attention_mask=mask,use_cache=False).logits[0,63].float()
        for j in range(steps):
            nx=int(lg.argmax()); toks.append(nx); gen.append(nx)
            buf[0,64+j]=nx; mask[0,64+j]=1
            lg=model(input_ids=buf,attention_mask=mask,use_cache=False).logits[0,64+j].float()
    return gen

res={}
for nm,fn in [("dup_cached",gen_dup),("clean_cached",gen_clean),("clean_padded",gen_padded)]:
    g=fn(m,toks0,256)
    res[nm]=stats(g)
    if ref_g.get("wiki_0"): res[nm]["==ref"]= res[nm]["sha"]==ref_g["wiki_0"]
    print(nm,res[nm],flush=True)
# HF-веса (свежие), чистый протокол
_hf=_o.environ.get("HF_HOME"); _o.environ["HF_HOME"]="/tmp/hfhf"
mh=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.float32).to("cuda").eval()
tkh=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
ids_h=tkh(pin["texts"][0],add_special_tokens=False)["input_ids"][:64]
res["hf_clean"]=stats(gen_clean(mh,ids_h,256))
res["hf_dup"]=stats(gen_dup(mh,ids_h,256))
res["hf_prompt_sha_same"]= sha(ids_h)==sha(list(map(int,toks0)))
print("hf:",res["hf_clean"],res["hf_dup"],"promptsha",res["hf_prompt_sha_same"],flush=True)
json.dump(res,open(f"{OUT}/GREEDY_PROTO.json","w"),indent=1,ensure_ascii=False)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)

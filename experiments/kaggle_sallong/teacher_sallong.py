# teacher_sallong.py — саліенс учителя 4B на длинных контекстах (32k) без материализации T×T.
# Механизм: хуки на q_norm/k_proj full-attn слоёв + rope-реконструкция; чанковый QK^T с top-k.
# Юнит-гейт (пре-рег): на T=1024 чанковый top-8 union по глубоким слоям пересекается с eager
# output_attentions union-ом не ниже 0.90 — иначе механизм не использовать.
import subprocess, sys, os, json, time
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2"])
import os as _o; _o.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
OUT="/kaggle/working"; _o.environ["HF_HOME"]="/tmp/hfs"
REV=None
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",dtype=torch.bfloat16,
    attn_implementation="eager").to("cuda").eval()
CFG=m.config
NL=CFG.num_hidden_layers
fullL=[i for i,t in enumerate(getattr(CFG,"layer_types",["full_attention"]*NL)) if t=="full_attention"]
if not fullL: fullL=list(range(NL))
NH=CFG.num_attention_heads; HD=getattr(CFG,"head_dim",CFG.hidden_size//NH)
CAP={}
def hk_qnorm(l):
    def h(mod,i,o): CAP[("q",l)]=o.detach()          # [1,T,NH*HD]
    return h
def hk_kproj(l):
    def h(mod,i,o): CAP[("k",l)]=o.detach()
    return h
for l in fullL:
    m.model.layers[l].self_attn.q_norm.register_forward_hook(hk_qnorm(l))
    m.model.layers[l].self_attn.k_proj.register_forward_hook(hk_kproj(l))
RPE={}
def hk_rope(mod,i,o):
    RPE["cs"],RPE["sn"]=(o[0].detach(),o[1].detach())
m.rotary_emb.register_forward_hook(hk_rope)

def rope_apply(x,cos,sin):
    # x [1,T,H*hd] -> [B? ; стандарт rotate_half qwen]
    B,T,N=x.shape; H=NH; hd=HD
    x=x.view(B,T,H,hd).transpose(1,2)                 # [B,H,T,hd]
    c=cos.unsqueeze(1); s=sin.unsqueeze(1)            # [B,1,T,hd]
    x1,x2=x[...,:hd//2],x[...,hd//2:]
    r1=x1*c[...,:hd//2]-x2*s[...,:hd//2]
    r2=x2*c[...,hd//2:]+x1*s[...,hd//2:]
    return torch.cat([r1,r2],-1)                      # [B,H,T,hd]

def sal_long(ids,chunk=2048,k=8,band="deep"):
    ids_t=torch.tensor([ids],device="cuda")
    CAP.clear()
    with torch.no_grad():
        m(input_ids=ids_t,use_cache=False)            # eager без output_attentions — только хуки
    T=len(ids)
    half=len(fullL)//2
    Ls=fullL[half:] if band=="deep" else (fullL if band=="full" else fullL[:half])
    union=torch.zeros(T,T,dtype=torch.bool,device="cuda")
    for l in Ls:
        q=CAP[("q",l)].float(); kp=CAP[("k",l)].float()
        qn=m.model.layers[l].self_attn  # norm уже применён (хук на q_norm)
        Q=rope_apply(q,RPE["cs"],RPE["sn"])          # [1,H,T,hd]
        K=rope_apply(kp,RPE["cs"],RPE["sn"])
        Q=Q[0]/ (HD**0.5)
        for c0 in range(0,T,chunk):
            c1=min(T,c0+chunk)
            S=Q[:,c0:c1,:]@K[0].transpose(1,2)       # [H,B,T]
            S[:,torch.arange(c1-c0,device="cuda")[:,None]+c0 < torch.arange(T,device="cuda")[None,:]]=float('-inf')
            ti=S.topk(min(k,T),dim=-1).indices        # [H,B,k]
            rows=(torch.arange(c0,c1,device="cuda"))
            for hh in range(S.shape[0]):
                idx=ti[hh]                            # [B,k]
                idx=torch.where(idx<=rows[:,None],idx,torch.tensor(-1,device="cuda"))
                union[rows[:,None].expand_as(idx)[idx>=0],idx[idx>=0]]=True
        del Q,K,S
        torch.cuda.empty_cache()
    # union->top8 на запрос: из bool-матрицы нельзя ранжировать; для AT нужен ранг:
    # сохраняем булево множество — байасу достаточно множества.
    del CAP; torch.cuda.empty_cache()
    return union.cpu().numpy()

def sal_eager(ids,k=8,band="deep"):
    with torch.no_grad():
        o=m(input_ids=torch.tensor([ids],device="cuda"),output_attentions=True,use_cache=False)
    T=len(ids); half=len(fullL)//2
    Ls=fullL[half:] if band=="deep" else fullL
    acc=torch.zeros(T,T,device="cuda")
    for l in Ls: acc+=o.attentions[l][0].float().mean(0)
    acc/=len(Ls)
    ti=torch.arange(T,device="cuda")
    acc=acc.masked_fill(ti[None,:]>ti[:,None],float('-inf'))
    idx=acc.topk(min(k,T),dim=1).indices
    ok=idx<=ti[:,None]
    u=np.zeros((T,T),dtype=bool)
    ii=np.arange(T)[:,None].repeat(k,1)[ok.cpu().numpy()]
    u[ii,idx.cpu().numpy()[ok.cpu().numpy()]]=True
    del o; torch.cuda.empty_cache()
    return u

# --- юнит-гейт на 1024 ---
import json as _j
pin_ok=True
text=("The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
      "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
      "modern AI were planted during the period called the Scientific Revolution, followed by a long "
      "series of discoveries that gave the world a new practical understanding of matter and energy. "*12)
ids=tk(text,add_special_tokens=False)["input_ids"][:1024]
a=sal_long(ids); b=sal_eager(ids)
row_ok=[]
for r in range(256,1024,32):
    ua=set(np.nonzero(a[r])[0]); ub=set(np.nonzero(b[r])[0])
    if ub: row_ok.append(len(ua&ub)/len(ub))
ov=float(np.mean(row_ok))
res={"unit_overlap":round(ov,3),"gate":bool(ov>=0.90),"T":1024,
     "full_layers":fullL,"nheads":NH,"head_dim":HD}
print(json.dumps(res,indent=1),flush=True)
json.dump(res,open(f"{OUT}/SALLONG_UNIT.json","w"),indent=1)
if ov>=0.90:
    # демонстрация на 8k (без AT, только механика/тайминг)
    t8=tk(text,add_special_tokens=False)["input_ids"][:8192]
    u=sal_long(t8); res["t8_ok"]=bool(u.any()); res["t8_density"]=float(u.mean())
    json.dump(res,open(f"{OUT}/SALLONG_UNIT.json","w"),indent=1)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)

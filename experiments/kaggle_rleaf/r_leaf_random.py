# r_leaf_random.py — R-тест на стороне чтения: лист (row space головы) на обученных vs случайных весах.
# Прогноз пре-рега (brainstorm v6 N-1 следствие 2): структурные скаляры листа близки к
# «архитектурным» на случайных весах (лист = row space головы по построению), в отличие от поля.
# Oценщик = exp45b (analytic_J + leaf_scalars), U128 = top-128 частотных токенов pin-текстов.
import subprocess, sys, os, json, time
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2"])
import numpy as np, torch, re
from collections import Counter
torch.set_grad_enabled(False)
IN="/kaggle/input"; OUT="/kaggle/working"
def find(slug,sub):
    for base in os.listdir(IN):
        if os.path.exists(os.path.join(IN,base,sub)): return os.path.join(IN,base)
    raise FileNotFoundError(slug)
W=find("qwen35-08b-weights","config.json"); P=find("prism-calib-pins-qwen","pins")
os.makedirs("/data/Qwen",exist_ok=True); tgt="/data/Qwen/Qwen3.5-0.8B"
if not os.path.exists(tgt): os.symlink(W,tgt)
from transformers import AutoModelForCausalLM, AutoTokenizer
m=AutoModelForCausalLM.from_pretrained(tgt,dtype=torch.float32,low_cpu_mem_usage=True).eval()
tk=AutoTokenizer.from_pretrained(tgt)
pin=json.load(open(os.path.join(P,"pins","wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")))
texts=pin["texts"][:12]
S=128; R=16; D=1024; SQD=D**0.5
# U128: top-128 подтокенов по частоте в pin-текстах
cnt=Counter()
for t in texts:
    cnt.update(tk(t,add_special_tokens=False)["input_ids"][:2048])
U128=[i for i,_ in cnt.most_common(128)]
print("U128 ready",len(U128),flush=True)

def hdoc23(model,ids):
    with torch.no_grad():
        hs=model.model(input_ids=ids.to("cpu"),output_hidden_states=True).hidden_states
    return hs[23][0].float()   # [S,D] layer-23 output, как L=23 в exp45b

def leaf_stats(model,ids):
    W_=model.lm_head.weight.detach().float()
    g=(1.0+model.model.norm.weight.detach().float())
    M=W_[U128]*g[None,:]                     # [K,D]
    H=hdoc23(model,ids)[:S]
    hn=H.norm(dim=-1).clamp_min(1e-12); u=H/hn[:,None]
    Mu=u@M.T
    J=(SQD/hn)[:,None,None]*(M[None,:,:]-Mu[:,:,None]*u[:,None,:])
    Jd=J.double()
    s2=torch.linalg.eigvalsh(Jd@Jd.transpose(1,2)).clamp_min(0.).flip(1); s=s2.sqrt()
    jn=s2.sum(1).sqrt()
    pr=(s2.sum(1)**2)/s2.pow(2).sum(1).clamp_min(1e-30)
    gu=(J.sum(1)**2).sum(1)/(128*(jn**2).clamp_min(1e-30))
    gap=((s[:,R-1]-s[:,R])/s[:,0].clamp_min(1e-30))
    return dict(j_norm=float(jn.median()),pr_sigma=float(pr.median()),g_uni=float(gu.median()),
                gap_r16=float(gap.median()),s16_over_s1=float((s[:,R-1]/s[:,0]).median()))

# тексты -> ids (перемые S токенов)
ids=[torch.tensor(tk(t,add_special_tokens=False)["input_ids"][:S],dtype=torch.long) for t in texts]
ids=[i for i in ids if len(i)==S]
trained=[leaf_stats(m,i) for i in ids[:4]]
print("trained done %.0fs"%(time.time()-t0),flush=True)
# случайные веса: все линейки normal(0,0.02), norm'ы как есть; тот же текстовый вход
import copy
state0={k:v.clone() for k,v in m.state_dict().items()}
res_rand=[]
for sd in range(3):
    g_=torch.Generator().manual_seed(1000+sd)
    with torch.no_grad():
        for k_,p_ in m.named_parameters():
            if "norm" in k_ or "ln" in k_ or k_.endswith(".bias"): continue
            p_.copy_(torch.normal(0.,0.02,size=tuple(p_.shape),generator=g_))
    res_rand.append([leaf_stats(m,i) for i in ids[:4]])
    print("rand seed",sd,"%.0fs"%(time.time()-t0),flush=True)
m.load_state_dict(state0)
med=lambda rows,c: float(np.median([r[c] for r in rows]))
cols=["j_norm","pr_sigma","g_uni","gap_r16","s16_over_s1"]
out={"trained":{c:med(trained,c) for c in cols},
     "random_by_seed":[{c:med(rr,c) for c in cols} for rr in res_rand],
     "U128":"top128-freq-in-pins","S":S,"protocol":"layer23 Hdoc; lm_head rows U128; exp45b estimator"}
print(json.dumps(out),flush=True)
json.dump(out,open(f"{OUT}/R_LEAF.json","w"),indent=1)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)

# kv_saliency.py — K-лестница (§21) H1-lite: салiенс-приор учителя (4B) как аддитивный байас
# внимания ученика (0.8B) ТОЛЬКО на full-attn слоях (pre-hook). Метрика: NLL held-out 96 токенов.
# Пре-рег: B1(deep) − B0 <= −0.02 nat, B2(shuf)≈B0, B3(other)≈B0 => H1 (адресация) подтверждена.
import subprocess, sys, os, json, time
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2","datasets==5.0.1"])
import os as _o; _o.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
IN="/kaggle/input"; OUT="/kaggle/working"
def find(name):
    for root,dirs,files in os.walk(IN):
        if name in files: return os.path.join(root,name)
        if name in dirs: return os.path.join(root,name)
    raise FileNotFoundError(name)
P=find("prism-calib-pins-qwen")
def pinof(pref):
    for root,dirs,files in os.walk(P):
        for f in files:
            if f.startswith(pref): return json.load(open(os.path.join(root,f)))
pin,pinm=pinof("wiki___data"),pinof("math___data")
_o.environ["HF_HOME"]="/tmp/hfk"
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
def windows(pinj,n,dom):
    out=[]
    for i,t in enumerate(pinj["texts"][:n]):
        ids=tk(t,add_special_tokens=False)["input_ids"][:448]
        if len(ids)>=200: out.append((f"{dom}{i}",ids))
    return out
DOCS=windows(pin,16,"w")+windows(pinm,16,"m")
print("docs:",len(DOCS),flush=True)

# ---------- фаза 1: учитель, saliency-кэш ----------
TE=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B",dtype=torch.bfloat16,
    attn_implementation="eager",output_attentions=True).to("cuda").eval()
NL=TE.config.num_hidden_layers
te_full=[i for i,t in enumerate(getattr(TE.config,"layer_types",["full_attention"]*NL)) if t=="full_attention"]
if not te_full: te_full=list(range(NL))
print("teacher full-attn:",len(te_full),"from",NL,flush=True)
def topk_rows(M,k=8):
    T=M.shape[0]; idx=torch.arange(T,device=M.device)
    M=M.masked_fill(idx[None,:]>idx[:,None],float('-inf'))
    ti=torch.topk(M,min(k,T),dim=1).indices
    r=idx[:,None].expand_as(ti)
    ok=ti<=r                                  # никакого будущего в байас
    ti=torch.where(ok,ti,torch.full_like(ti,-1))
    return ti.cpu().numpy()
def teacher_saliency(ids):
    with torch.no_grad():
        o=TE(input_ids=torch.tensor([ids],device="cuda"),output_attentions=True,use_cache=False)
    T=len(ids)
    half=len(te_full)//2
    deepL=[l for l in te_full if NL and te_full.index(l)>=half]
    shalL=[l for l in te_full if te_full.index(l)<half]
    def band(Ls):
        acc=torch.zeros(T,T,device="cuda")
        for l in Ls: acc+=o.attentions[l][0].float().mean(0)
        return acc/len(Ls)
    out={"deep":topk_rows(band(deepL)),"shallow":topk_rows(band(shalL)),"full":topk_rows(band(te_full))}
    del o; torch.cuda.empty_cache()
    return out
SAL={nm:teacher_saliency(ids) for nm,ids in DOCS}
print("saliency cached %.0fs"%(time.time()-t0),flush=True)
del TE
import gc; gc.collect(); torch.cuda.empty_cache()

# ---------- фаза 2: ученик + байас через pre-hook на full-attn ----------
ST=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.float32,
    attn_implementation="eager").to("cuda").eval()
st_full=[i for i,t in enumerate(getattr(ST.config,"layer_types",["full_attention"]*ST.config.num_hidden_layers)) if t=="full_attention"]
if not st_full: st_full=list(range(ST.config.num_hidden_layers))
CUR={"am":None}
def _mk_hook():
    def hk(mod,args,kwargs):
        if CUR["am"] is not None:
            kwargs=dict(kwargs); kwargs["attention_mask"]=CUR["am"]
        return args,kwargs
    return hk
for l in st_full: ST.model.layers[l].self_attn.register_forward_pre_hook(hk:=_mk_hook(),with_kwargs=True)
print("student full layers:",st_full,flush=True)
def student_nll(ids,bidx=None,alpha=0.0):
    T=len(ids); x=torch.tensor([ids],device="cuda")
    if bidx is None:
        CUR["am"]=None
    else:
        bi=torch.tensor(bidx,device="cuda")
        causal=torch.full((T,T),float('-inf'),device="cuda")
        causal=torch.where(torch.arange(T,device="cuda")[None,:]>torch.arange(T,device="cuda")[:,None],float('-inf'),0.0)
        am=causal.unsqueeze(0).unsqueeze(0).clone()
        rows=torch.arange(T,device="cuda")[:,None].expand_as(bi)
        ok=bi>=0
        am[0,0,rows[ok],bi[ok]]=alpha
        CUR["am"]=am
    with torch.no_grad():
        o=ST(input_ids=x,use_cache=False)
    CUR["am"]=None
    lg=o.logits[0,:-1].float(); tgt=torch.tensor(ids[1:],device="cuda")
    return float(F.cross_entropy(lg[-97:-1],tgt[-97:],reduction="mean"))
rng=np.random.default_rng(7)
res={}
def cell(name,key,alpha):
    vals=[]
    for di,(nm,ids) in enumerate(DOCS):
        T=len(ids)
        if key is None: v=student_nll(ids)
        else:
            bi=SAL[nm][key].copy()
            if key=="deep_shuf":
                flat=rng.permutation(bi.reshape(-1)).reshape(bi.shape)
                bi=flat
            elif key=="deep_other":
                onm,_=DOCS[(di+7)%len(DOCS)]
                ob=SAL[onm]["deep"]
                bi=ob[:T] if len(ob)>=T else np.pad(ob,((0,T-len(ob)),(0,0)),constant_values=-1)
            v=student_nll(ids,bi,alpha)
        vals.append(v)
    res[name]=round(float(np.median(vals)),4); print(name,res[name],"%.0fs"%(time.time()-t0),flush=True)
cell("B0_own",None,0.0)
cell("B1_deep","deep",3.0)
cell("B1_shallow","shallow",3.0)
cell("B1_full","full",3.0)
cell("B2_shuffled","deep_shuf",3.0)
cell("B3_other","deep_other",3.0)
cell("B1a1","deep",1.0)
res["meta"]=dict(docs=len(DOCS),topk=8,heldout=96,alphas="3.0 (1.0 для B1a1)",
    pre="B1-B0<=-0.02 и B2,B3 ~= B0 => H1")
json.dump(res,open(f"{OUT}/KV_SALIENCY.json","w"),indent=1)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)

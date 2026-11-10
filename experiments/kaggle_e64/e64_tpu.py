# e64_tpu.py — E-64 ядро на Kaggle TPU (torch_xla). Фазы:
#  P0 смоук/числовой гейт (dip wiki_0 ~ -0.07 +-0.015)
#  P1 own-greedy батарея (vec+hn+margin+p1+tok) 16 wiki
#  P2 corpus-scan клетка матрицы чтения (учительская размётка того же текста)
#  P3 R-null: переинициализация весов, 3 сида x 8 доков, слой 16
#  P4 T-лестница 1.2/1.5/2.0 x 16 доков (токены+маржины, кривые из скана vec)
#  P5 CPU-джобы: фазовый сдвиг resid, lead-lag, gap xdoc, lookup-coverage
import os, sys, json, time, traceback
os.environ["PJRT_DEVICE"]="TPU"
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]",*a,flush=True)
STATUS={}

import subprocess
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2","datasets==5.0.1"])
import numpy as np, torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

IN="/kaggle/input"; OUT="/kaggle/working"
def find(slug,sub):
    for base in os.listdir(IN):
        p=os.path.join(IN,base,sub)
        if os.path.exists(p): return os.path.join(IN,base)
    raise FileNotFoundError(slug)
PINS=find("prism-calib-pins-qwen","pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")
W=find("qwen35-08b-weights","config.json")
tgt="/data/Qwen"; os.makedirs("/data/Qwen",exist_ok=True)
if not os.path.exists(tgt): os.symlink(W,tgt)
os.environ["PRISM_CALIB_PIN_DIR"]=os.path.join(PINS,"pins")
os.environ["PRISM_CALIB_REQUIRE_PIN"]="1"
sys.path.insert(0,PINS)

# --- конфиг прогона (смоук vs полный задаётся флагом SMOKE) ---
SMOKE=os.environ.get("PRISM_SMOKE","1")=="1"
DOCS_OWN  = 2  if SMOKE else 16
DOCS_R    = 1  if SMOKE else 8
SEEDS_R   = 1  if SMOKE else 3
DOCS_T    = 2  if SMOKE else 16
STEPS     = 128 if SMOKE else 384
STEPS_T   = 128 if SMOKE else 256
MAXLEN    = 64+STEPS+32
LAYERS    = [1,5,11,17,23] if not SMOKE else [11,17,23]
log("SMOKE" if SMOKE else "FULL",DOCS_OWN,DOCS_R,DOCS_T,STEPS,LAYERS)

import calibration
DEV=xm.xla_device(); log("dev",DEV)
m=AutoModelForCausalLM.from_pretrained(tgt,torch_dtype=torch.float32).to(DEV).eval()
tk=AutoTokenizer.from_pretrained(tgt)
caps={}
def _mk(L):
    def _h(_mod,_inp,out):
        caps[L]=(out[0] if isinstance(out,tuple) else out)
    return _h
for L in LAYERS: m.model.layers[L].register_forward_hook(_mk(L))

def mark():
    xm.mark_step()

def fwd_buf(buf,mask,pos):
    """один forward фиксированной формы; возвращает (logit_row_np, {L: h_np}) на строке pos."""
    with torch.no_grad():
        o=m(input_ids=buf,attention_mask=mask,use_cache=False)
    mark()
    lg=o.logits[0,pos].detach().to("cpu",torch.float32).numpy()
    hs={L:caps[L][0,pos].detach().to("cpu",torch.float32).numpy() for L in LAYERS}
    return lg,hs

def run_doc(ids64,T,seed,steps):
    """no-cache фиксированной формы генерация; возвращает tok, vec[L] (steps,H), hn, margin, p1, same1."""
    torch.manual_seed(seed)
    buf=torch.full((1,MAXLEN),0,dtype=torch.long,device=DEV)
    mask=torch.zeros((1,MAXLEN),dtype=torch.long,device=DEV)
    buf[0,:64]=ids64.to(DEV); mask[0,:64]=1
    toks=ids64.tolist()
    vecL={L:[] for L in LAYERS}; hnL=[]; marg=[]; p1s=[]; same=[]
    lg,_=fwd_buf(buf,mask,63)
    h_prev=None
    for j in range(steps):
        p=torch.softmax(torch.tensor(lg),dim=-1).numpy()
        if T<=0: nxt=int(lg.argmax())
        else:
            pr=np.exp(lg/T); pr/=pr.sum(); nxt=int(np.random.choice(len(pr),p=pr))
        srt=np.sort(lg)[-2:]
        marg.append(float(lg.max()-srt[0])); p1s.append(float(p.max()))
        same.append(1.0 if toks and nxt==toks[-1] else 0.0)
        pos=63+j
        toks.append(nxt)
        buf[0,pos+1]=nxt; mask[0,pos+1]=1
        lg,hs=fwd_buf(buf,mask,pos+1)
        v=hs[LAYERS[-1]]
        hnL.append(float(np.linalg.norm(v)))
        for L in LAYERS: vecL[L].append(hs[L])
    tok=np.array(toks[64:64+steps],dtype=np.int64)
    vec={L:np.stack(vecL[L]) for L in LAYERS}
    return tok,vec,np.array(hnL),np.array(marg),np.array(p1s),np.array(same)

def curves_from_vec(V):
    """V [T,H] -> {dip,hump,plateau,curve k=1..48}: v_t=h_{t+1}-h_t, cos по t (оригинальный оценщик, без центрирования)."""
    Vn=V[1:]-V[:-1]
    D=np.linalg.norm(Vn,axis=1)
    D[D<1e-9]=1e-9
    Vn=Vn/D[:,None]
    out={}
    for k in range(1,49):
        if k>=len(Vn): break
        c=float(np.mean(np.sum(Vn[:-k]*Vn[k:],axis=1)))
        out[k]=c
    dip=out.get(1,float('nan'))
    ks=sorted(out); hump=max([out[k] for k in ks if 2<=k<=8],default=float('nan'))
    pl=[out[k] for k in ks if 9<=k<=24]
    return dict(dip=dip,hump=float(hump),plateau=float(np.mean(pl)) if pl else float('nan'),curve={str(k):out[k] for k in ks})

def gen_corpus(doc):
    """teacher-forced скан реального текста: один forward, vec по всем позициям текста."""
    ids=doc["input_ids"][0].tolist()[:448]
    n=len(ids)
    buf=torch.full((1,MAXLEN),0,dtype=torch.long,device=DEV)
    mask=torch.zeros((1,MAXLEN),dtype=torch.long,device=DEV)
    tt=torch.tensor(ids,dtype=torch.long,device=DEV)
    buf[0,:n]=tt; mask[0,:n]=1
    fwd_buf(buf,mask,n-1)
    full={L:caps[L][0,:n].detach().to("cpu",torch.float32).numpy() for L in LAYERS}
    return np.array(ids,dtype=np.int64),full

def lookup_cov(tok):
    """хеш-таблица: доля шагов, где предсказание по longest повт. суффиксу (<=16) совпало с выбором."""
    hits=0;n=0
    seen={}
    for t in range(1,len(tok)):
        pred=None
        for l in range(min(16,t-1),0,-1):
            key=tuple(tok[t-l:t])
            p=seen.get(key)
            if p is not None and p+1<len(tok) and p< t-l:
                pred=tok[p+1]; break
        if pred is not None and pred==tok[t]: hits+=1; n+=1
        elif pred is not None: n+=1
        # индексация
        for l in range(1,min(16,t)+1):
            seen.setdefault(tuple(tok[t-l+1:t+1]),t)
    return (hits/max(n,1),n)

def save_npz(name,**kw):
    np.savez_compressed(os.path.join(OUT,name),**kw)
    log("saved",name)

# =================== P0: данные + числовой гейт ===================
try:
    pools={dm:calibration.load_calibration_data(tgt,max_length=256,min_length=64,n_samples=16,domain=dm)
           for dm in ["wiki"]}
    ids0=pools["wiki"][0]["input_ids"][0,:64]
    tok,vec,hn,mg,p1,same=run_doc(ids0,0,47,STEPS)
    c16=curves_from_vec(vec[LAYERS[-1]])
    log("P0 dip",round(c16["dip"],4),"hump",round(c16["hump"],4),"p_same",round(float(np.mean(same)),3))
    STATUS["P0"]=dict(dip=c16["dip"],p_same=float(np.mean(same)),
        gate_abs=abs(c16["dip"]-(-0.0714))<0.02,gate_same=abs(float(np.mean(same))-0.606)<0.12,
        step_ms=(time.time()-T0)/ (STEPS+1) *1000)
    json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)
except Exception as e:
    STATUS["P0_ERR"]=repr(e); log("P0 ERR",e); traceback.print_exc()
    json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float); raise

if SMOKE:
    # быстрый smoke выход: фазы не гоняем, только тайминг
    log("SMOKE DONE"); json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float); raise SystemExit(0)

# =================== P1: own-greedy ===================
try:
    NPZ={}
    meta={}
    for si in range(DOCS_OWN):
        ids=pools["wiki"][si]["input_ids"][0,:64]
        tok,vec,hn,mg,p1,same=run_doc(ids,0,47000+si,STEPS)
        nm=f"own_w{si}"
        for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
        NPZ[f"{nm}_tok"]=tok; NPZ[f"{nm}_hn"]=hn; NPZ[f"{nm}_mg"]=mg; NPZ[f"{nm}_p1"]=p1
        cov=lookup_cov(tok.tolist())
        meta[nm]=dict(curves={str(L):curves_from_vec(vec[L]) for L in LAYERS},same1=float(same.mean()),lookup=cov)
        log("P1",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4),"cov",round(cov[0],3))
    save_npz("P1_own.npz",**NPZ); json.dump(meta,open(f"{OUT}/P1_own_meta.json","w"),default=float)
    STATUS["P1"]="ok"
except Exception as e:
    STATUS["P1_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P2: corpus cell ===================
try:
    NPZ={};meta={}
    for si in range(DOCS_OWN):
        ids,full=gen_corpus(pools["wiki"][si])
        nm=f"cor_w{si}"
        for L in LAYERS: NPZ[f"{nm}_L{L}"]=full[L][:len(ids)-1].astype(np.float16)  # vec-ряд состояний после токенов
        meta[nm]=dict(curves={str(L):curves_from_vec(full[L][:len(ids)-1].astype(np.float32)) for L in LAYERS})
        log("P2",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4),"nlen",len(ids))
    save_npz("P2_corpus.npz",**NPZ); json.dump(meta,open(f"{OUT}/P2_corpus_meta.json","w"),default=float)
    STATUS["P2"]="ok"
except Exception as e:
    STATUS["P2_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P3: R-null ===================
try:
    import copy
    state0={k:v.detach().to("cpu") for k,v in m.state_dict().items()}
    NPZ={};meta={}
    for sd in range(SEEDS_R):
        g=torch.Generator().manual_seed(1000+sd)
        with torch.no_grad():
            for k_,p in m.named_parameters():
                if "norm" in k_ or "ln" in k_ or k_.endswith(".bias"): continue
                p.copy_(torch.normal(0.,0.02,size=tuple(p.shape),generator=g))
        for si in range(DOCS_R):
            ids=pools["wiki"][si]["input_ids"][0,:64]
            tok,vec,hn,mg,p1,same=run_doc(ids,0,47000+si,STEPS)
            nm=f"rnull_s{sd}_w{si}"
            for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
            NPZ[f"{nm}_tok"]=tok
            meta[nm]=dict(curves={str(L):curves_from_vec(vec[L]) for L in LAYERS},same1=float(same.mean()))
            log("P3",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4),"same",round(float(same.mean()),3))
    m.load_state_dict({k:v.to(DEV) for k,v in state0.items()}) 
    save_npz("P3_rnull.npz",**NPZ); json.dump(meta,open(f"{OUT}/P3_rnull_meta.json","w"),default=float)
    STATUS["P3"]="ok"
except Exception as e:
    STATUS["P3_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P4: T-лестница 1.2/1.5/2.0 ===================
try:
    for T in [1.2,1.5,2.0]:
        NPZ={};meta={}
        for si in range(DOCS_T):
            ids=pools["wiki"][si]["input_ids"][0,:64]
            tok,vec,hn,mg,p1,same=run_doc(ids,T,78000+int(T*10)*100+si,STEPS_T)
            nm=f"T{T}_w{si}"
            for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
            NPZ[f"{nm}_tok"]=tok; NPZ[f"{nm}_mg"]=mg; NPZ[f"{nm}_p1"]=p1
            meta[nm]=dict(curves={str(L):curves_from_vec(vec[L]) for L in LAYERS},same1=float(same.mean()),
                          rep16=float(np.mean([len(set(tok[i:i+16]))<12 for i in range(0,len(tok)-16,16)])),
                          lookup=lookup_cov(tok.tolist())[0])
            log("P4",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4),"rep16",round(meta[nm]["rep16"],3))
        save_npz(f"P4_T{T}.npz",**NPZ); json.dump(meta,open(f"{OUT}/P4_T{T}_meta.json","w"),default=float)
        STATUS[f"P4_T{T}"]="ok"
except Exception as e:
    STATUS["P4_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P5: CPU-джобы по сохранённым vec ===================
try:
    res={}
    # фазовый сдвиг resid: V = smooth(G) + resid; сдвиг resid на +-48 -> плато умирает?
    z=np.load(os.path.join(OUT,"P1_own.npz"))
    Ls=str(LAYERS[-1])
    for si in range(min(4,DOCS_OWN)):
        V=z[f"own_w{si}_L{Ls}"].astype(np.float32)
        G=np.zeros_like(V)
        W=32
        for t in range(len(V)):
            a,b=max(0,t-W//2),min(len(V),t+W//2+1)
            G[t]=V[a:b].mean(0)
        R=V-G
        base=curves_from_vec(V)
        sh=dict()
        for s_ in [48,-48]:
            Vs=G+np.roll(R,s_,axis=0)
            sh[str(s_)]=curves_from_vec(Vs)
        mix=dict()
        for a_ in [0.0,0.5]:
            Vs=G+a_*R
            mix[str(a_)]=curves_from_vec(Vs)
        res[f"shift_w{si}"]=dict(base={k:base[k] for k in ["dip","hump","plateau"]},
            shift={k:{kk:v2[kk] for kk in ["dip","hump","plateau"]} for k,v2 in sh.items()},
            mix={k:{kk:v2[kk] for kk in ["dip","hump","plateau"]} for k,v2 in mix.items()})
        log("P5 shift w",si,res[f"shift_w{si}"]["base"]["plateau"],res[f"shift_w{si}"]["shift"]["48"]["plateau"])
    json.dump(res,open(f"{OUT}/P5_shift.json","w"),default=float)
    STATUS["P5"]="ok"
except Exception as e:
    STATUS["P5_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)
log("ALL DONE",STATUS)

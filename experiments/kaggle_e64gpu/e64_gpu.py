# e64_tpu.py — E-64 ядро на Kaggle TPU (torch_xla). Фазы:
#  P0 смоук/числовой гейт (dip wiki_0 ~ -0.07 +-0.015)
#  P1 own-greedy батарея (vec+hn+margin+p1+tok) 16 wiki
#  P2 corpus-scan клетка матрицы чтения (учительская размётка того же текста)
#  P3 R-null: переинициализация весов, 3 сида x 8 доков, слой 16
#  P4 T-лестница 1.2/1.5/2.0 x 16 доков (токены+маржины, кривые из скана vec)
#  P5 CPU-джобы: фазовый сдвиг resid, lead-lag, gap xdoc, lookup-coverage
import os, sys, json, time, traceback
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]",*a,flush=True)
STATUS={}

import subprocess
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2","datasets==5.0.1"])
import numpy as np, torch
USE_XLA=not torch.cuda.is_available()
if USE_XLA:
    os.environ["PJRT_DEVICE"]="TPU"
    import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

IN="/kaggle/input"; OUT="/kaggle/working"
def find(slug,sub):
    for root,dirs,files in os.walk(IN):
        if os.path.basename(root)==slug: return root
        if sub and sub in files: return os.path.dirname(root) if sub!=os.path.basename(sub) else root
    # fallback: поиск по имени sub-файла
    if sub:
        for root,dirs,files in os.walk(IN):
            if sub in files:
                return root if sub!=os.path.basename(sub) else os.path.dirname(root)
    raise FileNotFoundError(slug)
PINS_F=find("prism-calib-pins-qwen","wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")
PINS=os.path.dirname(PINS_F) if os.path.basename(PINS_F).startswith("wiki") else PINS_F
W=find("qwen35-08b-weights","config.json")
os.makedirs("/data/Qwen",exist_ok=True); tgt="/data/Qwen/Qwen3.5-0.8B"
if not os.path.exists(tgt): os.symlink(W,tgt)  # имя папки = ключ пина (calibration дерёт из basename)
os.environ["PRISM_CALIB_PIN_DIR"]=PINS  # датасет плоский: пины в корне
os.environ["PRISM_CALIB_REQUIRE_PIN"]="1"
sys.path.insert(0,PINS)
sys.path.insert(0,os.path.dirname(PINS))

# --- конфиг прогона (смоук vs полный задаётся флагом SMOKE) ---
SMOKE=os.environ.get("PRISM_SMOKE","0")=="1"  # мягкий якорь пройден в proto; полный прогон
DOCS_OWN  = 2  if SMOKE else 16
DOCS_R    = 1  if SMOKE else 8
SEEDS_R   = 1  if SMOKE else 3
DOCS_T    = 2  if SMOKE else 16
STEPS     = 128 if SMOKE else 384
STEPS_T   = 128 if SMOKE else 256
MAXLEN    = 64+STEPS+32
LAYERS    = [1,5,11,17,23] if not SMOKE else [11,17,23]
log("SMOKE" if SMOKE else "FULL",DOCS_OWN,DOCS_R,DOCS_T,STEPS,LAYERS)

import calibration, hashlib, zlib
def sha(x): return hashlib.sha256(repr(x).encode()).hexdigest()[:12]
REF=None
for root,dirs,files in os.walk(IN):
    if "exp47b_gen_kinematics.json" in files: REF=json.load(open(os.path.join(root,"exp47b_gen_kinematics.json")))
ref_p={f"{d['dom']}_{d['si']}":d["prompt_sha"] for d in REF["docs"]} if REF else {}
ref_g={f"{d['dom']}_{d['si']}":d["gen_sha"] for d in REF["docs"]} if REF else {}
def prompt_ids(pool_item):
    d=pool_item.get("data",pool_item)
    ids=d["input_ids"]
    if hasattr(ids,"dim") and ids.dim()==2: ids=ids[0]
    am=d.get("attention_mask")
    if am is not None:
        am=am[0] if hasattr(am,"dim") and am.dim()==2 else am
        nz=torch.nonzero(am).flatten()
        if len(nz): ids=ids[:int(nz[-1])+1]
    return list(map(int,ids))
DEV=torch.device("cuda") if torch.cuda.is_available() else xm.xla_device()
log("dev",DEV,"xla" if USE_XLA else "cuda")
m=AutoModelForCausalLM.from_pretrained(tgt,torch_dtype=torch.float32).to(DEV).eval()
tk=AutoTokenizer.from_pretrained(tgt)
caps={}
def _mk(L):
    def _h(_mod,_inp,out):
        caps[L]=(out[0] if isinstance(out,tuple) else out)
    return _h
for L in LAYERS: m.model.layers[L].register_forward_hook(_mk(L))

def mark():
    if USE_XLA: xm.mark_step()

def fwd_buf(buf,mask,pos):
    """один forward фиксированной формы; возвращает (logit_row_np, {L: h_np}) на строке pos."""
    with torch.no_grad():
        o=m(input_ids=buf,attention_mask=mask,use_cache=False)
    mark()
    lg=o.logits[0,pos].detach().to("cpu",torch.float32).numpy()
    hs={L:caps[L][0,pos].detach().to("cpu",torch.float32).numpy() for L in LAYERS}
    return lg,hs

def run_doc(ids64,T,seed,steps):
    """greedy/sampling с kv-кэшем (GPU-паттерн reshot); vec[hooks L], margin, p1, p_same."""
    import math
    torch.manual_seed(seed); np.random.seed(seed%2**32)
    from transformers import DynamicCache
    toks=list(map(int,ids64.tolist()))
    vecL={L:[] for L in LAYERS}; hnL=[]; marg=[]; p1s=[]; same=[]
    x=torch.tensor([toks],dtype=torch.long,device=DEV)
    with torch.no_grad():
        cache=DynamicCache(config=m.config)
        o=m(input_ids=x,past_key_values=cache,use_cache=True)
        lg=o.logits[0,-1].detach().float().cpu().numpy()
        mark()
        for j in range(steps):
            for L in LAYERS:
                h=caps[L][0,-1].detach().float().cpu().numpy(); vecL[L].append(h)
            p=np.exp(lg-lg.max()); p/=p.sum()
            if T<=0: nxt=int(lg.argmax())
            else: nxt=int(np.random.choice(len(p),p=p))
            srt=np.sort(lg)[-2:]
            marg.append(float(lg.max()-srt[0])); p1s.append(float(p.max()))
            same.append(1.0 if nxt==toks[-1] else 0.0)
            toks.append(nxt)
            with torch.no_grad():
                o=m(input_ids=torch.tensor([[nxt]],device=DEV),past_key_values=cache,use_cache=True)
                lg=o.logits[0,-1].detach().float().cpu().numpy()
                assert np.isfinite(lg).all()
                mark()
    tok=np.array(toks[len(toks)-steps:],dtype=np.int64)
    vec={L:np.stack(vecL[L]) for L in LAYERS}
    hnL=np.array([np.linalg.norm(vec[LAYERS[-1]][:,i]) for i in range(vec[LAYERS[-1]].shape[0])])
    return tok,vec,hnL,np.array(marg),np.array(p1s),np.array(same)

def dcurve(P,kk):
    """сериийный оценщик (exp64_b4_battery/A5): median cos по t, без средних."""
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def vpath_units(u):
    """сериийный Vpath: ортогонализованная составляющая шага между единичными состояниями."""
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    Vs=[]
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        if np.linalg.norm(vv)>1e-6: Vs.append(vv/np.linalg.norm(vv))
    return np.array(Vs)
def curves_series(Vh):
    """Vh [T,H] сырые h -> {dip,hump,plateau,curve{}} в ТОЧНОСТИ батареи B4 (median, ортогональн.шаг)."""
    Vh=Vh.astype(np.float64)
    u=Vh/np.maximum(np.linalg.norm(Vh,axis=1,keepdims=True),1e-12)
    P=vpath_units(u)
    kk=np.arange(1,min(96,max(2,len(P)//3))+1)
    c=dcurve(P,kk)
    return dict(dip=float(c[0]),hump=float(c[1:8].max()) if len(c)>8 else float('nan'),
        plateau=float(np.median(c[8:])) if len(c)>9 else float('nan'),
        curve={str(int(k)):float(x) for k,x in zip(kk,c)})

def gen_corpus(doc):
    """teacher-forced скан реального текста: один forward, vec по всем позициям текста."""
    ids=doc["input_ids"][0].tolist()[:min(448,MAXLEN-32)]
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
           for dm in ["wiki","math","code","chat"]}
    pr0=torch.tensor(prompt_ids(pools["wiki"][0]),dtype=torch.long)[:64]
    ok_p = (not ref_p) or sha(pr0.tolist())==ref_p["wiki_0"]
    tok,vec,hn,mg,p1,same=run_doc(pr0,0,47,STEPS)
    ok_g = (not ref_g) or sha(tok.tolist())==ref_g["wiki_0"]
    log("P0 sha prompt",ok_p,"gen",ok_g)
    c16=curves_series(vec[LAYERS[-1]])
    log("P0 dip",round(c16["dip"],4),"hump",round(c16["hump"],4),"p_same",round(float(np.mean(same)),3))
    STATUS["P0"]=dict(dip=c16["dip"],p_same=float(np.mean(same)),
        gate_abs=abs(c16["dip"]-(-0.0714))<0.03,gate_same=abs(float(np.mean(same))-0.606)<0.12,
        sha_prompt=ok_p,sha_gen=ok_g,
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
        ids=torch.tensor(prompt_ids(pools["wiki"][si]),dtype=torch.long)[:64]
        if ref_p: assert sha(ids.tolist())==ref_p[f"wiki_{si}"], f"pin mismatch w{si}"
        tok,vec,hn,mg,p1,same=run_doc(ids,0,47000+si,STEPS)
        nm=f"own_w{si}"
        for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
        NPZ[f"{nm}_tok"]=tok; NPZ[f"{nm}_hn"]=hn; NPZ[f"{nm}_mg"]=mg; NPZ[f"{nm}_p1"]=p1
        cov=lookup_cov(tok.tolist())
        meta[nm]=dict(curves={str(L):curves_series(vec[L]) for L in LAYERS},same1=float(same.mean()),lookup=cov)
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
        meta[nm]=dict(curves={str(L):curves_series(full[L][:len(ids)-1].astype(np.float32)) for L in LAYERS})
        log("P2",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4),"nlen",len(ids))
    save_npz("P2_corpus.npz",**NPZ); json.dump(meta,open(f"{OUT}/P2_corpus_meta.json","w"),default=float)
    STATUS["P2"]="ok"
except Exception as e:
    STATUS["P2_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P2c: cross-domain corpus (math/code/chat) ===================
try:
    NPZ={};meta={}
    for dm in ["math","code","chat"]:
        for si in range(min(8,DOCS_OWN)):
            ids,full=gen_corpus(pools[dm][si])
            nm=f"cor{dm[0]}_{si}"
            for L in LAYERS: NPZ[f"{nm}_L{L}"]=full[L][:len(ids)-1].astype(np.float16)
            meta[nm]=dict(curves={str(L):curves_series(full[L][:len(ids)-1]) for L in LAYERS})
            log("P2c",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4))
    save_npz("P2c_crossdom.npz",**NPZ); json.dump(meta,open(f"{OUT}/P2c_crossdom_meta.json","w"),default=float)
    STATUS["P2c"]="ok"
except Exception as e:
    STATUS["P2c_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P2b: foreign (тексты LFM3) ===================
try:
    FT=None
    for root,dirs,files in os.walk(IN):
        if "LFM3_TEXT.json" in files: FT=os.path.join(root,"LFM3_TEXT.json")
    if FT is None:
        STATUS["P2b"]="no LFM3_TEXT.json"
    else:
        ftxt=json.load(open(FT))
        NPZ={};meta={}
        for nm,tx in sorted(ftxt.items())[:DOCS_OWN]:
            ids=tk(tx,add_special_tokens=False)["input_ids"][:min(448,MAXLEN-32)]
            if len(ids)<64: continue
            buf=torch.full((1,MAXLEN),0,dtype=torch.long,device=DEV)
            mask=torch.zeros((1,MAXLEN),dtype=torch.long,device=DEV)
            buf[0,:len(ids)]=torch.tensor(ids,dtype=torch.long,device=DEV); mask[0,:len(ids)]=1
            fwd_buf(buf,mask,len(ids)-1)
            full={L:caps[L][0,:len(ids)].detach().to("cpu",torch.float32).numpy() for L in LAYERS}
            fnm="for_"+nm
            for L in LAYERS: NPZ[f"{fnm}_L{L}"]=full[L][:len(ids)-1].astype(np.float16)
            meta[fnm]=dict(curves={str(L):curves_series(full[L][:len(ids)-1]) for L in LAYERS})
            log("P2b",fnm,"dip",round(meta[fnm]["curves"][str(LAYERS[-1])]["dip"],4),"nlen",len(ids))
        save_npz("P2b_foreign.npz",**NPZ); json.dump(meta,open(f"{OUT}/P2b_foreign_meta.json","w"),default=float)
        STATUS["P2b"]="ok"
except Exception as e:
    STATUS["P2b_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P3: R-null ===================
state0=None
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
            ids=torch.tensor(prompt_ids(pools["wiki"][si]),dtype=torch.long)[:64]
            tok,vec,hn,mg,p1,same=run_doc(ids,0,47000+si,STEPS)
            nm=f"rnull_s{sd}_w{si}"
            for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
            NPZ[f"{nm}_tok"]=tok
            meta[nm]=dict(curves={str(L):curves_series(vec[L]) for L in LAYERS},same1=float(same.mean()))
            log("P3",nm,"dip",round(meta[nm]["curves"][str(LAYERS[-1])]["dip"],4),"same",round(float(same.mean()),3))
    save_npz("P3_rnull.npz",**NPZ); json.dump(meta,open(f"{OUT}/P3_rnull_meta.json","w"),default=float)
    STATUS["P3"]="ok"
except Exception as e:
    STATUS["P3_ERR"]=repr(e); traceback.print_exc()
finally:
    if state0 is not None:
        m.load_state_dict({k:v.to(DEV) for k,v in state0.items()})
        mark(); STATUS["MODEL_RESTORED"]=True
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)

# =================== P4: T-лестница 1.2/1.5/2.0 ===================
try:
    for T in [1.2,1.5,2.0]:
        NPZ={};meta={}
        for si in range(DOCS_T):
            ids=torch.tensor(prompt_ids(pools["wiki"][si]),dtype=torch.long)[:64]
            tok,vec,hn,mg,p1,same=run_doc(ids,T,78000+int(T*10)*100+si,STEPS_T)
            nm=f"T{T}_w{si}"
            for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
            NPZ[f"{nm}_tok"]=tok; NPZ[f"{nm}_mg"]=mg; NPZ[f"{nm}_p1"]=p1
            meta[nm]=dict(curves={str(L):curves_series(vec[L]) for L in LAYERS},same1=float(same.mean()),
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
    # фазовый сдвиг resid в ПРОСТРАНСТВЕ A5: V=vpath(u); G=MA(V,16); сдвиг R=V-G на ±48
    z=np.load(os.path.join(OUT,"P1_own.npz"))
    Ls=str(LAYERS[-1]); KM=np.arange(1,25)
    def a5(V):
        u=V.astype(np.float64); u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
        return vpath_units(u)
    def ma(X,w=16):
        from numpy.lib.stride_tricks import sliding_window_view
        pad=np.pad(X,(((w-1)//2,w//2),(0,0)),mode="edge")
        return sliding_window_view(pad,w,axis=0).mean(-1)  # [T,D] ровно
    for si in range(min(4,DOCS_OWN)):
        V=a5(z[f"own_w{si}_L{Ls}"])
        G=ma(V); R=V-G
        def stat(X):
            c=dcurve(X,KM); return dict(dip=float(c[0]),hump=float(c[1:8].max()),plateau=float(np.median(c[8:])))
        res[f"shift_w{si}"]=dict(base=stat(V),
            shift={str(s_):stat(G+np.roll(R,s_,axis=0)) for s_ in [48,-48]},
            mix={str(a_):stat(G+a_*R) for a_ in [0.0,0.5]})
        log("P5 shift w",si,res[f"shift_w{si}"]["base"]["plateau"],res[f"shift_w{si}"]["shift"]["48"]["plateau"])
    json.dump(res,open(f"{OUT}/P5_shift.json","w"),default=float)
    STATUS["P5"]="ok"
except Exception as e:
    STATUS["P5_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)
log("ALL DONE",STATUS)

# e64d_64docs.py — E-76d: те же клетки P1(own-greedy)/P2(corpus), но 64 окна вместо 16.
# Цель (ЭРРАТА E-76c): split-half floor при n=16 слишком широк (L1 [0.26..0.89]) —
# «общая λ₁» не отличима от «разных». n=64 сужает floor ~1/n. Только две фазы, LAYERS=[1,11,23].
# Дисайн окон: 16 пинов (как в E-64) + по 3 дополнительных окна на тот же wiki-док
# (свежая токенизация 0.8B, смещения 64+64*w, w=1..3) — тексты разные, домены те же.
# greedy-сиды НОВЫЕ (48000+si): собственные walk'и, не переиспользование P1.
# Запуск на Kaggle TPU так же, как e64_gpu.py (датасеты: pins + weights + calibration.py рядом).
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
        if sub and sub in files: return root if sub!=os.path.basename(sub) else os.path.dirname(root)
    if sub:
        for root,dirs,files in os.walk(IN):
            if sub in files: return root if sub!=os.path.basename(sub) else os.path.dirname(root)
    raise FileNotFoundError(slug)
PINS_F=find("prism-calib-pin-qwen","wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")
PINS=os.path.dirname(PINS_F)
W=find("qwen35-08b-weights","config.json")
os.makedirs("/data/Qwen",exist_ok=True); tgt="/data/Qwen/Qwen3.5-0.8B"
if not os.path.exists(tgt): os.symlink(W,tgt)
os.environ["PRISM_CALIB_PIN_DIR"]=PINS; os.environ["PRISM_CALIB_REQUIRE_PIN"]="1"
sys.path.insert(0,PINS); sys.path.insert(0,os.path.dirname(PINS))
LAYERS=[1,11,23]
STEPS=64 if os.environ.get("PRISM_SMOKE","0")=="1" else 384   # геометрия клетки как в E-64 (STEPS=384)
MAXLEN=64+STEPS+32
import calibration
pools={"wiki":[d for d in calibration.load_calibration_data("Qwen/Qwen3.5-4B",max_length=256,
        min_length=64,n_samples=16,domain="wiki")]}
tk4=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
tk=AutoTokenizer.from_pretrained(tgt)
def prompt_ids(pool_item):
    d=pool_item.get("data",pool_item); ids=d["input_ids"]
    if hasattr(ids,"dim") and ids.dim()==2: ids=ids[0]
    am=d.get("attention_mask")
    if am is not None:
        am=am[0] if hasattr(am,"dim") and am.dim()==2 else am
        nz=torch.nonzero(am).flatten()
        if len(nz): ids=ids[:int(nz[-1])+1]
    return list(map(int,ids))
# 64 окна: w0 = пин (тексты пина, токенизация 0.8B), w1..3 = хвостовые срезы декода
WINDOWS=[]  # (name, ids64, is_pin)
pin=json.load(open(os.path.join(PINS,"wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")))
assert len(pin["texts"])==16
for si in range(16):
    ids=tk(pin["texts"][si],max_length=256,truncation=True,add_special_tokens=False)["input_ids"][0].tolist()[:256]
    WINDOWS.append((f"w{si}_pin",ids[:64],True))
    dec=tk4.decode(prompt_ids(pools["wiki"][si]))
    for w in (1,2,3):
        extra=tk(dec[40*w:40*w+260],add_special_tokens=False,return_tensors="pt")["input_ids"][0]
        extra=extra.tolist()[:64]
        if len(extra)>=48: WINDOWS.append((f"w{si}_x{w}",extra,False))
log("windows",len(WINDOWS))
DEV=torch.device("cuda") if torch.cuda.is_available() else xm.xla_device()
m=AutoModelForCausalLM.from_pretrained(tgt,torch_dtype=torch.float32).to(DEV).eval()
caps={}
def _mk(L):
    def _h(_mod,_inp,out): caps[L]=(out[0] if isinstance(out,tuple) else out)
    return _h
for L in LAYERS: m.model.layers[L].register_forward_hook(_mk(L))
def mark():
    if USE_XLA: xm.mark_step()
def fwd_buf(buf,mask,pos):
    with torch.no_grad(): o=m(input_ids=buf,attention_mask=mask,use_cache=False)
    mark()
    lg=o.logits[0,pos].detach().to("cpu",torch.float32).numpy()
    hs={L:caps[L][0,pos].detach().to("cpu",torch.float32).numpy() for L in LAYERS}
    return lg,hs
# run_doc — ДОСЛОВНО из e64_gpu.py (kv-кэш; T<=0 ⇒ greedy argmax — семантику не трогать!)
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
src=open(os.path.join(os.path.dirname(__file__),"e64_gpu.py")).read() if os.path.exists(os.path.join(os.path.dirname(__file__),"e64_gpu.py")) else ""
def save_npz(name,**kw):
    torch.save(kw,os.path.join(OUT,name))
# ==== P1x: own-greedy 64 окна ====
try:
    NPZ={}; meta={}
    for si,(nm,ids,is_pin) in enumerate(WINDOWS):
        ids64=torch.tensor(ids,dtype=torch.long)[:64]
        tok,vec,hn,mg,p1,same=run_doc(ids64,0,48000+si,STEPS)   # T=0 ⇒ greedy, как E-64
        for L in LAYERS: NPZ[f"{nm}_L{L}"]=vec[L].astype(np.float16)
        NPZ[f"{nm}_tok"]=tok; NPZ[f"{nm}_hn"]=hn
        meta[nm]=dict(pin=bool(is_pin),n=len(tok))
        log("P1x",nm,"pin" if is_pin else "extra","len",len(tok))
    save_npz("P1_own64.npz",**NPZ); json.dump(meta,open(f"{OUT}/P1_own64_meta.json","w"),default=float)
    STATUS["P1x"]="ok"
except Exception as e:
    STATUS["P1x_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)
# ==== P2x: corpus-скан тех же 64 окон ====
try:
    NPZ={}; meta={}
    for si,(nm,ids,is_pin) in enumerate(WINDOWS):
        n=len(ids)
        buf=torch.full((1,MAXLEN),0,dtype=torch.long,device=DEV)
        mask=torch.zeros((1,MAXLEN),dtype=torch.long,device=DEV)
        buf[0,:n]=torch.tensor(ids,device=DEV); mask[0,:n]=1
        fwd_buf(buf,mask,n-1)
        for L in LAYERS:
            full=caps[L][0,:n].detach().to("cpu",torch.float32).numpy()
            NPZ[f"{nm}_L{L}"]=full[:n-1].astype(np.float16)   # состояния ПОСЛЕ токенов, как P2
        meta[nm]=dict(pin=bool(is_pin),nlen=n)
        log("P2x",nm,"nlen",n)
    save_npz("P2_corpus64.npz",**NPZ); json.dump(meta,open(f"{OUT}/P2_corpus64_meta.json","w"),default=float)
    STATUS["P2x"]="ok"
except Exception as e:
    STATUS["P2x_ERR"]=repr(e); traceback.print_exc()
json.dump(STATUS,open(f"{OUT}/STATUS.json","w"),default=float)
print("E64d DONE",STATUS,flush=True)

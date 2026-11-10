# b4_reshot.py — ре-семпл T-таблицы B4 (аудит §19.4): T∈{0.3,0.7,1.0} × 32 дока (wiki+math),
# greedy-loop с kv-кэшем, сохранение per-doc: tok, vec L23, margin, p1, p_same + сериийные кривые.
# Тот оценщик, что в серии (median, ортогональный шаг). Фиксирует якоря для TPU-лестницы 1.2–2.0.
import subprocess, sys, os, json, time
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2","datasets==5.0.1"])
import os as _o; _o.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import numpy as np, torch
IN="/kaggle/input"; OUT="/kaggle/working"
def find(name):
    for root,dirs,files in os.walk(IN):
        if name in files: return os.path.join(root,name)
        if name in dirs: return os.path.join(root,name)
    raise FileNotFoundError(name)
W=find("qwen35-08b-weights"); PINS=find("prism-calib-pins-qwen")
tgt="/data/Qwen/Qwen3.5-0.8B"; os.makedirs("/data/Qwen",exist_ok=True)
if not os.path.exists(tgt): os.symlink(W,tgt)
sys.path.insert(0,PINS); sys.path.insert(0,os.path.join(PINS,"pins"))
_o.environ["PRISM_CALIB_PIN_DIR"]=os.path.join(PINS,"pins")
_o.environ["PRISM_CALIB_REQUIRE_PIN"]="1"
import calibration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache
m=AutoModelForCausalLM.from_pretrained(tgt,dtype=torch.bfloat16).to("cuda").eval()
L_SET=[23]; caps={}
def _mk(L):
    def _h(_mod,_inp,out):
        caps[L]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu().numpy()
    return _h
for L in L_SET: m.model.layers[L].register_forward_hook(_mk(L))
def dcurve(P,kk):
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])
def vpath_units(u):
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12); Vs=[]
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        if np.linalg.norm(vv)>1e-6: Vs.append(vv/np.linalg.norm(vv))
    return np.array(Vs)
def curves_series(Vh):
    Vh=Vh.astype(np.float64); u=Vh/np.maximum(np.linalg.norm(Vh,axis=1,keepdims=True),1e-12)
    P=vpath_units(u); kk=np.arange(1,97)
    c=dcurve(P,kk)
    return dict(dip=float(c[0]),hump=float(c[1:8].max()),plateau=float(np.median(c[8:])),
                curve={str(int(k)):float(x) for k,x in zip(kk,c)})
pools={dm:calibration.load_calibration_data(tgt,max_length=256,min_length=64,n_samples=16,domain=dm)
       for dm in ["wiki","math"]}
STEPS=384
def run_doc(ids64,T,seed):
    torch.manual_seed(seed); np.random.seed(seed%2**32)
    toks=ids64.tolist(); vec=[]; marg=[]; p1s=[]; same=[]
    ids=torch.tensor([toks],device="cuda")
    with torch.no_grad():
        cache=DynamicCache(config=m.config)
        m(input_ids=ids,past_key_values=cache,use_cache=True)
        h=caps[23].copy()
        for j in range(STEPS):
            vec.append(h)
            with torch.no_grad():
                lg=m(input_ids=torch.tensor([[toks[-1]]],device="cuda"),past_key_values=cache,use_cache=True).logits[0,-1].float().cpu().numpy()
            p=np.exp(lg-lg.max()); p/=p.sum()
            if T<=0: nxt=int(lg.argmax())
            else: nxt=int(np.random.choice(len(p),p=p))
            srt=np.sort(lg)[-2:]; marg.append(float(lg.max()-srt[0])); p1s.append(float(p.max()))
            same.append(1.0 if nxt==toks[-1] else 0.0)
            toks.append(nxt)
            with torch.no_grad():
                m(input_ids=torch.tensor([[nxt]],device="cuda"),past_key_values=cache,use_cache=True)
            h=caps[23].copy()
    return np.array(toks[64:64+STEPS],np.int64),np.stack(vec),np.array(marg),np.array(p1s),np.array(same)
NPZ={};meta={}
for T in [0.3,0.7,1.0]:
    for di,dm in enumerate(["wiki","math"]):
        for si in range(16):
            ids=pools[dm][si]["input_ids"][0,:64]
            tok,vec,mg,p1,same=run_doc(ids,T,77000+int(T*10)*1000+di*16+si)
            nm=f"T{T}_{dm}_{si}"
            NPZ[f"{nm}_L23"]=vec.astype(np.float16); NPZ[f"{nm}_tok"]=tok
            NPZ[f"{nm}_mg"]=mg; NPZ[f"{nm}_p1"]=p1
            c=curves_series(vec)
            meta[nm]=dict(dom=dm,T=T,curves={"23":c},same1=float(same.mean()))
            if si%4==3: print(nm,"dip",round(c["dip"],4),"%.0fs"%(time.time()-t0),flush=True)
    np.savez_compressed(f"{OUT}/RESHOT_T{T}.npz",**NPZ)
    json.dump(meta,open(f"{OUT}/RESHOT_T{T}_meta.json","w"),default=float)
    print("saved T",T,"%.0fs"%(time.time()-t0),flush=True); NPZ={}
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)

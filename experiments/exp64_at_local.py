#!/usr/bin/env python
"""AT-генерации локально (CPU fp32): 6 контекстов × 32 дока (si 0..7), greedy 384,
u/hn L{1,5,11,17,23}. Схема/сиды/хуки — копия Kaggle-ядра. Прогресс по доку."""
import json, os, sys, time, hashlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"6")
sys.path.insert(0,'/home/survivor/work/prism')
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0,'/home/survivor/work/prism/prism')
os.environ["PRISM_CALIB_PIN_DIR"]="/home/survivor/work/prism/data/calib_pins"
os.environ["PRISM_CALIB_REQUIRE_PIN"]="1"
import calibration as cal
def sha(x): return hashlib.sha256(repr(x).encode()).hexdigest()[:12]
t0=time.time()
tgt="/data/Qwen/Qwen3.5-0.8B"
m=AutoModelForCausalLM.from_pretrained(tgt,dtype=torch.float32).eval()
Lset=[1,5,11,17,23]; caps={}; clg={}
for L in Lset:
    m.model.layers[L].register_forward_hook(lambda _md,_i,o,L=L: caps.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[0,-1].detach().double()))
m.lm_head.register_forward_hook(lambda _md,_i,o: clg.__setitem__("g",o[0,-1].detach()))
CTX=json.load(open("/tmp/at/contexts/AT_contexts.json"))
pools={dm:cal.load_calibration_data(tgt,max_length=256,min_length=64,n_samples=16,domain=dm)
       for dm in ["wiki","math","code","chat"]}
print("loaded",time.time()-t0,flush=True)
def gen_doc(ids,plen):
    toks=ids[0].tolist(); uL={L:[] for L in Lset}; hnL={L:[] for L in Lset}
    with torch.no_grad():
        cache=DynamicCache(config=m.config)
        m(input_ids=ids,past_key_values=cache,use_cache=True)
    h={L:caps[L].clone() for L in Lset}; logits=clg["g"].clone()
    with torch.no_grad():
        for step in range(384):
            for L in Lset:
                uL[L].append((h[L]/h[L].norm()).half().cpu()); hnL[L].append(float(h[L].norm()))
            nxt=int(logits.argmax()); toks.append(nxt)
            m(input_ids=torch.tensor([[nxt]]),past_key_values=cache,use_cache=True)
            h={L:caps[L].clone() for L in Lset}; logits=clg["g"].clone()
    return toks[plen:],uL,hnL
OUT="/data/prism_kernels/exp62"
for cond in [k for k in CTX if k.startswith("C")]:
    outf=f"{OUT}/gen_AT_{cond}_qwen.npz"
    if os.path.exists(outf): print("skip",cond,flush=True); continue
    ctx=torch.tensor([CTX[cond]],dtype=torch.long); npz={}; meta={}
    for dm in ["wiki","math","code","chat"]:
        for si in range(8):
            nm=f"{dm}_{si}"
            ids=torch.cat([ctx,pools[dm][si]["input_ids"][:,:64]],dim=1)
            gen,uL,hnL=gen_doc(ids,CTX["T"]+64)
            for L in Lset:
                npz[f"{nm}_L{L}_u"]=np.stack([x.numpy() for x in uL[L]]); npz[f"{nm}_L{L}_hn"]=np.array(hnL[L],dtype=np.float32)
            npz[f"{nm}_tok"]=np.array(gen,dtype=np.int32); meta[nm]={"dom":dm,"si":si,"cond":cond,"gen_sha":sha(gen)}
            print(cond,nm,f"{time.time()-t0:.0f}s",flush=True)
    np.savez_compressed(outf,**npz)
    json.dump({"cond":cond,"ctxT":CTX["T"],"docs":meta},open(outf.replace(".npz","_meta.json"),"w"))
    print("saved",cond,time.time()-t0,flush=True)
print("ALL DONE",time.time()-t0,flush=True)

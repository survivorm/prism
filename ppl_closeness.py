#!/usr/bin/env python
"""Ковариаты контекстов AT: PPL (понятность) и closeness (близость к собственному регистру модели).
closeness = средний cos L23-состояний контекста к облаку u-состояний T0-генераций модели.
Только CPU, fp32, один forward на контекст."""
import json, os, sys, torch, numpy as np
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
sys.path.insert(0,"/home/survivor/work/prism")
from transformers import AutoModelForCausalLM
dev="cpu"
m=AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B",dtype=torch.float32).eval()
CTX=json.load(open("/tmp/at/contexts/AT_contexts.json"))
z=np.load("/data/prism_kernels/exp62/gen_layers_qwen.npz")
meta=json.load(open("/data/prism_kernels/exp62/gen_layers_qwen_meta.json"))
pool=[]
for n in list(meta)[:24]:
    u=z[f"{n}_L23_u"].astype(np.float32)
    pool.append(u[::7]); 
pool=np.concatenate(pool); pool/=np.maximum(np.linalg.norm(pool,axis=1,keepdims=True),1e-9)
out={}
for c in [k for k in CTX if k.startswith("C")]:
    ids=torch.tensor([CTX[c]])
    with torch.no_grad():
        o=m(input_ids=ids,output_hidden_states=True)
    lg=o.logits[:,:-1].float(); lb=ids[:,1:]
    nll=torch.nn.functional.cross_entropy(lg.reshape(-1,lg.shape[-1]),lb.reshape(-1),reduction="mean")
    h=o.hidden_states[24][0].numpy().astype(np.float32); h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-9)
    cs=h@pool.T
    out[c]={"ppl":round(float(np.exp(nll)),2),"close_mean":round(float(cs.mean()),4),
            "close_p90":round(float(np.percentile(cs,90)),4)}
    print(c,out[c],flush=True)
json.dump(out,open("/tmp/at/ctx_covariates.json","w"),indent=1)
print("saved covariates")

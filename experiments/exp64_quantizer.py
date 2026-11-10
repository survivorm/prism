#!/usr/bin/env python
"""E-64a/b: квантователь без форвардов (E64_pre тесты 1–3).
a) Δ(1) из ОДНИХ эмбеддингов: Δe_t = e(T_{t+1})−e(T_t); кривая cos(Δe_t,Δe_{t+k}).
   Если ≈ real −0.072/−0.084 — дип это статистика языка, модель ни при чём.
   Плюс p_same(d), d=1..4 (period-2 в токенах).
b) «колонка глубины»: g_l(t)=h^{l'}(t)−h^l(t) при фиксированном токене —
   межтокенная устойчивость нити cos(g(t),g(t+1)) и её лаг-кривая (несёт ли нить ω*)."""
import json, os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np

NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"
MODEL="/data/Qwen/Qwen3.5-0.8B"; EMB_KEY="model.language_model.embed_tokens.weight"
z=np.load(NPZ, allow_pickle=True); meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
names=sorted(meta,key=lambda n:(["wiki","math","code","chat"].index(dom[n]),int(n.rsplit("_",1)[-1])))

# --- embedding matrix (только один тензор, без модели) ---
import glob
try:
    idx=json.load(open(f"{MODEL}/model.safetensors.index.json")); fn=idx["weight_map"][EMB_KEY]
    from safetensors.numpy import load_file
    EMB=load_file(f"{MODEL}/{fn}")[EMB_KEY]
except FileNotFoundError:
    from safetensors.numpy import load_file
    f0=glob.glob(f"{MODEL}/*.safetensors")[0]; EMB=load_file(f0)[EMB_KEY]
print("EMB:",EMB.shape,EMB.dtype,flush=True)

KM=24
def curve(V,kmax=KM):
    Vn=V/np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Vn[:-k],Vn[k:]))) for k in range(1,kmax+1)])
rng=np.random.default_rng(11)
LAY=[("1","5"),("5","11"),("11","17"),("17","23")]
emb={}; shuf={}; depth={}; ps={n:{} for n in names}
for i,nm in enumerate(names):
    tok=np.asarray(z[f"{nm}_tok"],dtype=np.int64)
    if i==0: print("tok len:",len(tok),flush=True)
    Eb=EMB[tok].astype(np.float64)
    emb[nm]=curve(np.diff(Eb,axis=0))
    shuf[nm]=np.mean([curve(np.diff(EMB[tok[rng.permutation(len(tok))]].astype(np.float64),axis=0)) for _ in range(3)],axis=0)
    for d in range(1,5):
        ps[nm][d]=float(np.mean(tok[:-d]==tok[d:]))
    h={L:np.asarray(z[f"{nm}_L{L}_u"],dtype=np.float64)*np.asarray(z[f"{nm}_L{L}_hn"],dtype=np.float64)[:,None] for L in ["1","5","11","17","23"]}
    dd={}
    for a,b in LAY:
        g=h[b]-h[a]; gn=g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
        dd[f"{a}->{b}"]={"c":curve(g).tolist(),"c1next":float(np.mean(np.einsum("ij,ij->i",gn[:-1],gn[1:])))}
    depth[nm]=dd
    if i%16==15: print("...",i+1,flush=True)

def dm(f,d,key=None):
    ns=[n for n in names if dom[n]==d]
    if key is None: return np.median(np.array([f[n] for n in ns]),axis=0)
    return float(np.median([f[n][key] for n in ns]))
out={"emb":{"per_doc":{n:list(np.round(emb[n],4)) for n in names},
            "by_dom":{d:dm(emb,d).round(4).tolist() for d in ["wiki","math","code","chat"]},
            "shuf_by_dom":{d:dm(shuf,d).round(4).tolist() for d in ["wiki","math","code","chat"]}},
     "p_same":{d:{str(k):float(np.median([ps[n][k] for n in names if dom[n]==d])) for k in range(1,5)} for d in ["wiki","math","code","chat"]},
     "depth":{ "c1next_by_dom_pair":{f"{a}->{b}":{d:dm({n:depth[n][f"{a}->{b}"]["c1next"] for n in names},d) for d in ["wiki","math","code","chat"]} for a,b in LAY},
               "curve_by_dom_pair":{f"{a}->{b}":{d:np.median(np.array([[depth[n][f"{a}->{b}"]["c"] for n in names if dom[n]==d][0]]),axis=0).round(4).tolist() for d in ["wiki","math","code","chat"]} for a,b in LAY}}}
print("\n=== a) Δ(1)_emb по доменам (real батарея −0.072; T-17 V-оценщик per-dom −0.096/−0.088/−0.066/−0.096) ===")
for d in ["wiki","math","code","chat"]:
    print(f"{d:5s} emb k=1..6:",[round(x,3) for x in out["emb"]["by_dom"][d][:6]],
          " shuf k=1:",round(out["emb"]["shuf_by_dom"][d][0],3),
          " p_same 1..4:",[round(out["p_same"][d][str(k)],3) for k in range(1,5)])
print("\n=== b) нить: cos(g(t),g(t+1)) по парам слоёв (wiki,math,code,chat) ===")
for a,b in LAY:
    r=out["depth"]["c1next_by_dom_pair"][f"{a}->{b}"]
    print(f"{a}->{b:2s}:", {d:round(r[d],3) for d in r})
json.dump(out,open("../results/e64/e64ab_quantizer.json","w"))
print("saved e64ab_quantizer.json")
